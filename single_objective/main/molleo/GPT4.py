from openai import OpenAI
import re
from rdkit import Chem
import main.molleo.crossover as co, main.molleo.mutate as mu
import random
import os

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MINIMUM = 1e-10

def query_LLM(question, model="gpt-4.1-mini", temperature=0.0):
    message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]

    prompt1 = question
    message.append({"role": "user", "content": prompt1})

    for retry in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                temperature=temperature,
                messages=message
            )
            response_content = response.choices[0].message.content
            message.append({"role": "assistant", "content": response_content})
            break
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            if retry == 2:  # Last retry
                raise e

    print("=>")
    return message, response_content

class GPT4:
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
                }
        self.task2objective = {
                'qed': 'Please propose a new molecule that has a higher QED score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'drd2': 'Please propose a new molecule that has a higher DRD2 score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
                'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You should make crossover and mutations based on the given molecules, using your molecular knowledge to guide the modifications.\n\n',
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
            
            # Add LLM-driven mutation instructions
            mutation_instructions = """\n\nMolecular Design Strategy:
            - Analyze the parent molecules and identify structural features that contribute to their scores
            - Create a new molecule through intelligent crossover and mutation of the parent structures
            - Focus on functional group modifications, ring changes, side chain alterations, and stereochemical changes
            - Use your molecular knowledge to guide modifications that are likely to improve the target property
            - Ensure all proposed molecules remain chemically valid and synthesizable
            - Aim for meaningful structural diversity while building upon successful molecular features. Ensure that the molecule is valid in structure and can be parsed.\n\n"""
                        
            prompt = task_definition + mol_tuple + task_objective + mutation_instructions + self.requirements
            _, r = query_LLM(prompt)
            
            # Extract explanation and molecule
            explanation = ""
            try:
                explanation_match = re.search(r'<<<Explaination>>>:\s*(.*?)<<<Molecule>>>', r, re.DOTALL)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    # Fallback: use the full response as explanation if format not found
                    explanation = r
            except:
                explanation = r
            
            proposed_smiles = re.search(r'\\box\{(.*?)\}', r).group(1)
            proposed_smiles = sanitize_smiles(proposed_smiles)
            print(proposed_smiles)
            assert proposed_smiles != None
            new_child = Chem.MolFromSmiles(proposed_smiles)

            # Return molecule, explanation, and prompt as a tuple
            return new_child, explanation, prompt
        except Exception as e:
            print(f"{type(e).__name__} {e}")
            # Fallback: Always do crossover + mutation since LLM should handle all mutations
            new_child = co.crossover(parent_mol[0], parent_mol[1])
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            else:
                # If crossover fails, just mutate one parent
                new_child = mu.mutate(random.choice(parent_mol), mutation_rate)
            
            # Return molecule with fallback explanation and prompt
            fallback_explanation = f"Fallback genetic crossover + mutation due to LLM error: {e}"
            fallback_prompt = f"Fallback genetic operation for parents: {[Chem.MolToSmiles(mol) for mol in parent_mol]}"
            return new_child, fallback_explanation, fallback_prompt
    
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