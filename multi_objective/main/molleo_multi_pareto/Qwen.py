import requests
import json
import re
from rdkit import Chem
import main.molleo_multi_pareto.crossover as co, main.molleo_multi_pareto.mutate as mu
import random

MINIMUM = 1e-10

# vLLM server configuration
VLLM_SERVER_URL = "http://localhost:8000/v1/chat/completions"

def query_LLM(question, temperature=0.0):
    """Query the Qwen model using vLLM server"""
    
    # Create the chat messages
    messages = [
        {"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."},
        {"role": "user", "content": question}
    ]
    
    # Prepare the request payload
    payload = {
                    "model": "Qwen/Qwen2.5-7B-Instruct",  # This should match the model served by vLLM
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "stop": ["User:", "System:"]
    }
    
    for retry in range(3):
        try:
            # Make HTTP request to vLLM server
            response = requests.post(
                VLLM_SERVER_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"].strip()
            
            # Create message format for compatibility
            messages.append({"role": "assistant", "content": assistant_response})
            
            print("=>")
            return messages, assistant_response
            
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            if retry == 2:  # Last retry
                print("Failed to connect to vLLM server. Make sure it's running on localhost:8000")
                raise e
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            if retry == 2:  # Last retry
                raise e

    return None, None

class Qwen:
    def __init__(self):
        self.task2description = {
                'qed': 'I have two molecules and their QED scores. The QED score measures the drug-likeness of the molecule.\n\n',
                'jnk3': 'I have two molecules and their JNK3 scores. The JNK3 score measures a molecular\'s biological activity against JNK3.\n\n',
                'drd2': 'I have two molecules and their DRD2 scores. The DRD2 score measures a molecule\'s biological activity against a biological target named the dopamine type 2 receptor (DRD2).\n\n',
                'gsk3b': 'I have two molecules and their GSK3$\beta$ scores. The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'I have two molecules and their isomer scores. The isomer score measures a molecule\'s similarity in terms of atom counter to C9H10N2O2PF2Cl.\n\n',
                'perindopril_mpo': 'I have two molecules and their perindopril multiproperty objective scores. The perindopril multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to perindopril and number of aromatic rings.\n\n',
                'sitagliptin_mpo': 'I have two molecules and their sitagliptin multiproperty objective scores. The sitagliptin rediscovery score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to sitagliptin, TPSA score, LogP score and isomer score with C16H15F6N5O.\n\n',
                'ranolazine_mpo': 'I have two molecules and their ranolazine multiproperty objective scores. The ranolazine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to ranolazine, TPSA score LogP score and number of fluorine atoms.\n\n',
                'thiothixene_rediscovery': 'I have two molecules and their thiothixene rediscovery measures a molecule\'s Tanimoto similarity with thiothixene\'s SMILES to check whether it could be rediscovered.\n\n',
                'mestranol_similarity': 'I have two molecules and their mestranol similarity scores. The mestranol similarity score measures a molecule\'s Tanimoto similarity with Mestranol.\n\n',
                'sa': 'I have two molecules and their SA scores. The SA score measures the synthetic accessibility of the molecule.\n\n',
                }
        self.task2objective = {
                'qed': 'Please propose a new molecule that has a higher QED score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'drd2': 'Please propose a new molecule that has a higher DRD2 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'sa': 'Please propose a new molecule that has a lower SA score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                }
        self.requirements = """\n\nYour output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n
        \n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid.
        """
        self.task=None

    def edit(self, mating_tuples, mutation_rate):
        task = self.task
        task_definition = self.task2description[task[0]]
        task_objective = self.task2objective[task[0]]

        parent = []
        parent.append(random.choice(mating_tuples))
        parent.append(random.choice(mating_tuples))
        parent_mol = [t[1] for t in parent]
        parent_scores = [t[0] for t in parent]
        try:
            mol_tuple = ''
            for i in range(2):
                tu = '\n[' + Chem.MolToSmiles(parent_mol[i]) + ',' + str(parent_scores[i]) + ']'
                mol_tuple = mol_tuple + tu
            prompt = task_definition + mol_tuple + task_objective + self.requirements
            _, r = query_LLM(prompt)
            
            # Extract explanation and molecule
            explanation_match = re.search(r'<<<Explaination>>>:\s*(.*?)(?=<<<Molecule>>>|$)', r, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            
            proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
            proposed_smiles = sanitize_smiles(proposed_smiles)
            
            print(f"  Explanation: {explanation}")
            print(f"  Generated SMILES: {proposed_smiles}")
            
            assert proposed_smiles != None
            new_child = Chem.MolFromSmiles(proposed_smiles)

            return new_child
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            new_child = co.crossover(parent_mol[0], parent_mol[1])
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            return new_child
    
def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None 