#!/usr/bin/env python3

import re

# Sample response from the user's data
test_response = """1. {<<<Explanation>>>: Inspired by the top molecule #1, which has a protonated amine linked to a sulfonamide attached to a fused bicyclic heteroaromatic system, I designed a molecule that retains the protonated amine and sulfonamide but replaces the bicyclic system with a quinoline-thiazole hybrid scaffold to add aromatic nitrogen heterocycles, likely improving binding and electronic properties. The protonated amine and sulfonamide are known to enhance solubility and interaction, and the fused heterocycles should increase affinity. This crossover between #1 and the heterocycles seen in #3 and #15 aims to increase reward beyond 0.30.  
<<<Molecule>>>: CCC[NH2+]CCNS(=O)(=O)c1ccc2c(n1)nc3scccc23}}

2. {<<<Explanation>>>: Combining the chalcone-type structure with methoxy substituents from #2 and the imide heterocycle from #7 and #20, I designed a molecule where a dimethoxyphenyl chalcone is linked to a phthalimide moiety. The conjugated enone system provides planarity and potential Michael acceptor properties, while the imide may enhance binding and metabolic stability. This molecule is a hybrid of #2 and #7/#20, aiming to improve potency and reward (above 0.30) with good drug-like features.  
<<<Molecule>>>: COc1ccc(/C=C/C2=O)cc(OC)c1C=CC(=O)N3C(=O)c4ccccc4C3=O}}"""

def test_extraction():
    print("Testing direct pair extraction...")
    
    # Find all explanation+molecule pairs directly
    pattern = r'<<<Explanation>>>:\s*(.*?)<<<Molecule>>>:\s*([^\s\}]+)'
    matches = re.findall(pattern, test_response, re.DOTALL)
    
    print(f"Found {len(matches)} explanation+molecule pairs")
    
    for i, (explanation, smiles) in enumerate(matches, 1):
        print(f"\n--- Pair {i} ---")
        
        # Clean up
        explanation = explanation.strip()
        explanation = re.sub(r',\s*$', '', explanation)
        smiles = smiles.strip()
        
        print(f"✅ Explanation: {explanation[:80]}...")
        print(f"✅ SMILES: {smiles}")

if __name__ == "__main__":
    test_extraction() 