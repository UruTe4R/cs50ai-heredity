import csv
import itertools
import sys
import math

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    names = set(people)
    no_gene = names - one_gene - two_genes # set of names with no gene
    not_have_trait = names - have_trait# set of names without trait
    
    bios = {person: {"gene": 0} for person in no_gene}
    bios.update({person: {"gene": 1} for person in one_gene})
    bios.update({person: {"gene": 2} for person in two_genes})
    p = []
    
    # Find origin parents
    for name in names:
        person = people[name]#set of person's bio
        if person["mother"] is None and person["father"] is None:

            if person["name"] in no_gene: 
                if person["name"] in have_trait:
                    p.append(PROBS["trait"][0][True] * PROBS["gene"][0])
                elif person["name"] in not_have_trait:
                    p.append(PROBS["trait"][0][False] * PROBS["gene"][0])

            elif person["name"] in one_gene: 
                if person["name"] in have_trait:
                    p.append(PROBS["gene"][1] * PROBS["trait"][1][True])
                elif person["name"] in not_have_trait:
                    p.append(PROBS["gene"][1] * PROBS["trait"][1][False])

            elif person["name"] in two_genes:
                if person["name"] in have_trait:
                    p.append(PROBS["gene"][2] * PROBS["trait"][2][True])
                elif person["name"] in not_have_trait:
                    p.append(PROBS["gene"][2] * PROBS["trait"][2][False])

            
        elif person["mother"] is not None and person["father"] is not None:

            # get mother and father's name
            m_n = person["mother"]
            f_n = person["father"]
            genes_m = bios[m_n]["gene"]
            genes_f = bios[f_n]["gene"]
            mut = PROBS["mutation"]

            if person["name"] in no_gene:# both parents passed no gene
                if person["name"] in have_trait:
                    p.append(PROBS["trait"][0][True])
                else:
                    p.append(PROBS["trait"][0][False])

                if genes_m == 0:

                    # p_m & p_f = probability of mother/father passing genes or no gene
                    p_m = 1 - mut
                    p_f = 0
                    if genes_f == 0:
                        p_f = 1 - mut
                    elif genes_f == 1:
                        p_f = 0.5
                    elif genes_f == 2:
                        p_f = mut
                    p.append(p_m * p_f)

                elif genes_m == 1:
                    p_m = 0.5
                    p_f = 0
                    if genes_f == 0:
                        p_f = 1 - mut
                    elif genes_f == 1:
                        p_f = 0.5
                    elif genes_f == 2: 
                        p_f = mut
                    p.append(p_m * p_f)

                elif genes_m == 2:
                    p_m = mut
                    p_f = 0
                    if genes_f == 0:
                        p_f = 1 - mut
                    elif genes_f == 1:
                        p_f = 0.5
                    elif genes_f == 2:
                        p_f = mut
                    p.append(p_m * p_f)

            elif person["name"] in one_gene:# inherited one gene
                if person["name"] in have_trait:
                    p.append(PROBS["trait"][1][True])
                else:
                    p.append(PROBS["trait"][1][False])
                if genes_m == 0:

                    if genes_f == 0:
                        # mutation happens to mother or father and the other parant passes no gene
                        p.append(2 * (mut * (1 - mut)))
                    elif genes_f == 1:    
                        # mother mutated and father didn't pass or mother didnot pass and father passed                      
                        p.append((0.5 * mut + 0.5 * (1 - mut)))
                    elif genes_f == 2:
                        # mutation happens to both or 0 gene from mother and 1 gene from father
                        p.append((mut * mut) + (1 - mut) * (1 - mut))
                
                elif genes_m == 1:

                    if genes_f == 0:
                        p.append((0.5 * mut + 0.5 * (1 - mut)))
                    elif genes_f == 1:
                        p.append(2 * (0.5 * 0.5))
                    elif genes_f == 2:
                        p.append(0.5 * mut + 0.5 * (1 - mut)) 
                
                elif genes_m == 2:

                    if genes_f == 0:
                        p.append((1 - mut)**2 + mut**2)
                    elif genes_f == 1:
                        p.append((1 - mut) * 0.5 + mut * 0.5)
                    elif genes_f == 2:
                        p.append(2 * (1 - mut) * mut)

            elif person["name"] in two_genes:#inherited 2 genes
                if person["name"] in have_trait:
                    p.append(PROBS["trait"][2][True])
                else:
                    p.append(PROBS["trait"][2][False])
                if genes_m == 0:
                    p_m = mut
                    p_f = 0
                    if genes_f == 0:
                        p_f = mut
                    elif genes_f == 1:
                        p_f = 0.5
                    elif genes_f == 2:
                        p_f = 1 - mut
                    p.append(p_m * p_f)
                elif genes_m == 1:
                    p_m = 0.5
                    p_f = 0
                    if genes_f == 0:
                        p_f = mut
                    elif genes_f == 1:
                        p_f = 0.5
                    elif genes_f == 2:
                        p_f = 1 - mut
                    p.append(p_m * p_f)
                elif genes_m == 2:
                    p_m = 1 - mut
                    p_f = 0
                    if genes_f == 0:
                        p_f = mut
                    elif genes_f == 1:
                        p_f = 0.5
                    elif genes_f == 2:
                        p_f = 1 - mut
                    p.append(p_m * p_f)
        
    p = [round(prob, 5) for prob in p]
    return math.prod(p)

            



def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    
    names = set(probabilities)
    no_gene = names - one_gene - two_genes
    not_have_trait = names - have_trait
    bios = {name: {"gene": 0} for name in no_gene}
    bios.update({name: {"gene": 1} for name in one_gene})
    bios.update({name: {"gene": 2} for name in two_genes})



    for name in names:
        if name in have_trait:
            probabilities[name]["trait"][True] += p
        elif name in not_have_trait:
            probabilities[name]["trait"][False] += p

        if name in no_gene:
            probabilities[name]["gene"][0] += p 
        elif name in one_gene:
            probabilities[name]["gene"][1] += p
        elif name in two_genes:
            probabilities[name]["gene"][2] += p
    


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    names = probabilities.keys()
    for name in names:
        total_gene = sum(probabilities[name]["gene"].values())
        probabilities[name]["gene"][0] = probabilities[name]["gene"][0] / total_gene
        probabilities[name]["gene"][1] = probabilities[name]["gene"][1] / total_gene
        probabilities[name]["gene"][2] = probabilities[name]["gene"][2] / total_gene
        total_trait = sum(probabilities[name]["trait"].values())
        probabilities[name]["trait"][True] = probabilities[name]["trait"][True] / total_trait
        probabilities[name]["trait"][False] = probabilities[name]["trait"][False] / total_trait


if __name__ == "__main__":
    main()
