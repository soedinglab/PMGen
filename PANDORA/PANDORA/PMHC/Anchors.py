from PANDORA import Contacts

def pMHCI_anchors(pdb):
    ''' Finds peptide anchor residues of p:MHCI complexes

    Args:
        pdb: Bio.PDB object (or path to pdb file (string))

    Returns: (list of ints) Peptide anchor residue positions

    '''

    def closest_pept_res(dist_list):
        '''Find the closest peptide residue based on te minimal mean value between peptide residues and MHC residues'''
        anch_list = []
        for i in set([i[2] for i in dist_list]):
            l = [x[-1] for x in dist_list if x[2] == i]
            anch_list.append((i, sum(l) / len(l))) #add mean distance between pept residue and MHC residue
        return min(((x[1], x[0]) for x in anch_list), default=(0, 0))[1] #get the pept residue with the min mean dist

    # Rafaella's residues
    # pocket_M = {'anch1': [24, 25, 35, 36], 'anch2': [81]}

    # Derek's modified version. Empirically tested to have the highest prediction accuracy. See below. (only best shown)
    pocket_M = {'anch1': [7, 24, 35], 'anch2': [118, 135]}

    # Accuracies peptide residue anchor prediction using MHCI residue specified above.
    # I selected all structures with non-canonically spaced anchors from Rafaella's prediction and 100 random pdbs.
    # From this dataset I manually determined the peptide anchor residue position. I then looped through all MHCI
    # residues and selected the ones with the highest accuracy in predicting the correct anchors.

    # Getting all anchors right --> Accuracy: 100.0
    # Anchor 1: 100.0
    # Anchor 2: 100.0

    # Calculate the contacts with a cutoff of 18. This cutoff because at lower cutoffs, the orientations of side chains
    # of different structures, result in highly different contacts.
    cont = Contacts.Contacts(pdb, pept_contacts=True, M_only=sum(pocket_M.values(), []), cutoff=25).chain_contacts

    # Make sure the first chain is P, second chain is M or N, not the other way around
    cont = cont + [(i[4], i[5], i[6], i[7], i[0], i[1], i[2], i[3], i[8]) for i in cont if i[1] == 'P' or i[5] == 'P']
    cont = list(dict.fromkeys(cont))

    # Find the peptide residues that reside in the M and N pockets
    anch1, anch2 = [], []
    for i in cont:
        if i[1] == 'P' and i[5] == 'M': # Use residues from the M chain to find anchor 1, 3 and 4
            # Check the Alpha, Beta and Gamma carbon molecules of the peptide residues.
            if i[3] == 'CA' or i[3] == 'CB' or i[3] == 'CG' or i[3] == 'C1':
                if i[6] in pocket_M['anch1']: #If connected (<25 ang) to specified anch 1 MHC residues, add to list
                    anch1.append(i)
                if i[6] in pocket_M['anch2']: #If connected (<25 ang) to specified anch 2 MHC residues, add to list
                    anch2.append(i)

    # Now find the peptide residues that are closest to the specified MHCII residues to get the anchors
    anchor_1 = closest_pept_res(anch1)
    anchor_2 = closest_pept_res(anch2)
    # Put the anchors in the right order if they aren't already
    anchors = sorted([i for i in [anchor_1, anchor_2]])

    # Return the ordered set of anchors. Filter out 0's and >25 numbers if they happen to get through.
    return sorted(set([i for i in anchors if i != 0 or i > 25]))


def pMHCII_anchors(pdb):
    ''' Finds peptide anchor residues of p:MHCII complexes

    Args:
        pdb: Bio.PDB object (or path to pdb file (string))

    Returns: (list of ints) Peptide anchor residue positions

    '''

    def closest_pept_res(dist_list):
        '''Find the closest peptide residue based on te minimal mean value between peptide residues and MHC residues'''
        anch_list = []
        for i in set([i[2] for i in dist_list]):
            l = [x[-1] for x in dist_list if x[2] == i]
            anch_list.append((i, sum(l) / len(l))) #add mean distance between pept residue and MHC residue
        return min(((x[1], x[0]) for x in anch_list), default=(0, 0))[::-1] #get the pept residue with the min mean dist

    # Rafaella's residues
    # pocket_M = {'anch1':[28, 33, 31], 'anch3':[10]}
    # pocket_N = {'anch2':[10, 11, 21, 22, 23], 'anch3':[6,7], 'anch4':[ 28, 27, 31, 33]}

    # Derek's modified version. Empirically tested to have the highest prediction accuracy.
    # Has a 100% accuracy on all IMGT templates
    pocket_M = {'anch1': [28, 29], 'anch2': [82, 81, 80], 'anch4': [82, 81, 80], 'anch3': [113, 114]}
    pocket_N = {'anch2': [24, 25], 'anch3': []}

    cont = Contacts.Contacts(pdb, pept_contacts=True, M_only=sum(pocket_M.values(), []),
                             N_only=sum(pocket_N.values(), []), cutoff=25).chain_contacts

    # Make sure the first chain is P, second chain is M or N, not the other way around
    cont = cont + [(i[4], i[5], i[6], i[7], i[0], i[1], i[2], i[3], i[8]) for i in cont if i[1] == 'P' or i[5] == 'P']
    cont = list(dict.fromkeys(cont))


    pept_len = max([i[2] for i in cont if i[1] == 'P'])

    # Find the peptide residues that reside in the M and N pockets
    anch1, anch2, anch3, anch4 = [], [], [], []
    for i in cont:
        if i[1] == 'P' and i[5] == 'M': # Use residues from the M chain to find anchor 1, 3 and 4
            # Check the Alpha, Beta and Gamma carbon molecules of the peptide residues.
            if i[3] == 'CA' or i[3] == 'CB' or i[3] == 'CG':
                if i[6] in pocket_M['anch1']: #If connected (<25 ang) to specified anch 1 MHC residues, add to list
                    anch1.append(i)
                if i[6] in pocket_M['anch2']: #If connected (<25 ang) to specified anch 2 MHC residues, add to list
                    anch2.append(i)
                if i[6] in pocket_M['anch3']: #If connected (<25 ang) to specified anch 3 MHC residues, add to list
                    anch3.append(i)
                if i[6] in pocket_M['anch4']: #If connected (<25 ang) to specified anch 4 MHC residues, add to list
                    anch4.append(i)

        if i[1] == 'P' and i[5] == 'N': # Use chain N for the second anchor
            # Check the Alpha, Beta and Gamma carbon molecules of the peptide residues
            if i[3] == 'CA' or i[3] == 'CB' or i[3] == 'CG':
                if i[6] in pocket_N['anch2']: #If connected (<25 ang) to specified anch 2 MHC residues, add to list
                    anch2.append(i)

    # First check if the peptide is inverted: C to N term instead of N to C term (this is very rare)
    anchor_1 = closest_pept_res(anch1)
    extra_rules = True
    # If the first anchor is very deep inside the 1st pocket, we can be pretty confident in assigning the other anchors
    if anchor_1[1] < 18:

        # find the distance between res M51 and the predicted first anchor. Make sure its less than 15 angstrom.
        cont_p1 = Contacts.Contacts(pdb, pept_contacts=True, M_only=[51],
                                    N_only=[], cutoff=12).chain_contacts
        # Make sure the first chain is P, second chain is M or N, not the other way around
        cont_p1 = cont_p1 + [(i[4], i[5], i[6], i[7], i[0], i[1], i[2], i[3], i[8]) for i in cont_p1 if
                       i[1] == 'P' or i[5] == 'P']
        cont_p1 = list(dict.fromkeys(cont_p1))
        # Check the Alpha, Beta and Gamma carbon molecules of the peptide residues.
        cont_p1 = [i for i in cont_p1 if i[1] == 'P' and i[5] == 'M' and (i[3] == 'CA' or i[3] == 'CB' or i[3] == 'CG')]

        # Find the distance between residue M51 and the predicted 1st anchor
        dist_between_51_anch1 = closest_pept_res([i for i in cont_p1 if i[2] == anchor_1[0]])[1]

        # If predicted anchor 1 is very close to res M51, then we can be very sure to have found pept res anchor 1
        # deep and firmly inside in the pocket --> assume canonical spacing then.
        if dist_between_51_anch1 != 0 and dist_between_51_anch1 < 12:

            anchor_1 = anchor_1[0]
            anchor_2 = anchor_1 + 3
            anchor_3 = anchor_1 + 5
            if pept_len - anchor_1 >= 8:
                anchor_4 = anchor_1 + 8
            else:
                anchor_4 = 0
            extra_rules = False

    # If its not, use some extra rules to increase prediction accuracy.
    if extra_rules:
        # First find anchor 2, this anchor is found with the highest accuracy
        if pept_len >= 9:
            anch2 = [i for i in anch2 if i[2] < pept_len - 4]
        anchor_2 = closest_pept_res(anch2)[0]

        # Make sure anchor 1 if smaller than anchor 2, but bigger than anchor 2 - 5
        # This makes sure residues from long flanking regions that are flipped over the MHC structure aren't selected
        anch1_mod = [i for i in anch1 if i[2] < anchor_2 - 1 and i[2] > anchor_2 - 5]
        anchor_1 = closest_pept_res(anch1_mod)[0]

        # Make sure anchor 3 is close to anchor 2 as well (but not too far, to prevent selecting anchor 4)
        anch3_mod = [i for i in anch3 if i[2] > anchor_2 + 1 and i[2] < anchor_2 + 4]
        anchor_3 = closest_pept_res(anch3_mod)[0]

        # Same for anchor 4. Make sure its not too far away. Otherwise distant flanking residues might get selected.
        anch4_mod = [i for i in anch4 if i[2] > anchor_2 + 2 and i[2] < anchor_2 + 6]
        anchor_4 = closest_pept_res(anch4_mod)[0]

    # Put the anchors in the right order if they aren't already
    anchors = sorted([i for i in [anchor_1, anchor_2, anchor_3, anchor_4]])

    # If the spacing between 1-4 and 1-2 is both wrong, infer anchors positions based on anchor 1
    if anchors[-1] - anchors[0] != 8 and anchors[1] - anchors[0] != 3:
        anchor_1 = closest_pept_res(anch1)[0]
        if pept_len - anchor_1 >= 8:
            anchors = [anchor_1, anchor_1 + 3, anchor_1 + 5, anchor_1 + 8]
        else:
            anchors = [anchor_1, anchor_1 + 3, anchor_1 + 5, 0]

    # If anchor 1 and 4 are spaced in the canonical way, and anchor 2 and 3 are not, then just jump the gun and
    # use the canonical spacing for anchor 2 and 3.
    if anchors[-1] - anchors[0] == 8:
        if anchors[1] - anchors[2] != -2 or anchors[2] - anchors[1] != 2:
            anchors[1] = anchors[0] + 3
            anchors[2] = anchors[0] + 5

    # If there is the 3rd and 4th anchor are adjacent and the 4th anchor is the last peptide, remove 4th anchor
    if anchors[3] - anchors[2] == 1 and pept_len == anchors[3]:
            anchors[3] = 0

    # Return the ordered set of anchors. Filter out 0's and >25 numbers if they happen to get through.
    return sorted(set([i for i in anchors if i != 0 or i > 25]))
