"""
    # usage: pdbfilepath A H,L Yes

"""
import os,sys
import numpy as np

def parse_pdb(pdbfile):
    """return content,header
    """
    with open(pdbfile,'r') as fin:
        fullcontent = [l for l in fin]
    
    header = [l for l in fullcontent if l.startswith('REMARK')]
    content = [l for l in fullcontent if l.startswith('ATOM')]
    # content = [l for l in content if l.count('UNK') == 0] # remove UNKOWN residues
    return np.array(content)

def chain_splitter(content,chains_AG,chains_AB):
    """
    Return the content that corresponds to the chains_molecule
    """
    chaini = np.array([l[21] for l in content])
    mask_AG = np.isin(chaini, np.array(chains_AG))
    mask_AB = ~mask_AG
    return content[mask_AG], content[mask_AB]

def rechain_renumber(lines,chain,resstart,atomstart):
    newlines = []
    prev_resid = None
    resid = resstart
    ai = atomstart
    for line in lines:
        ai += 1
        line_resuid = line[17:27]
        if line_resuid != prev_resid:
            prev_resid = line_resuid
            resid += 1
            if resid > 9999:
                print('Cannot set residue number above 9999.')
                sys.exit(1)
        
        newlines.append('ATOM{:>7}{}{}{:>4}{}'.format(ai,line[11:21],chain,resid,line[26:]))
    ter_record = 'TER {:>7}      {} {}{:>4}                                                      \n'.format(ai,newlines[-1][17:20],chain,resid)
    newlines.append(ter_record)
    return newlines, resid, len(newlines) -1

def write_model(new_content,output_path):
    with open(output_path,'w') as fout:
        for l in new_content:
            fout.write(l)
    

def organizer(pdbfile,chains_AG,chains_AB,output_path):

    # parse the PDB
    content = parse_pdb(pdbfile)

    # split the identified chains
    cag, cab = chain_splitter(content,chains_AG,chains_AB)
    if len(cag) == 0:
        print(f'Did not detect antigen chains {chains_AG} in pdbfile {pdbfile}')
        return []
    if len(cab) == 0:
        print(f'Did not detect antibody chains {chains_AB} in pdbfile {pdbfile}')
        return []
    # rechain and renumber
    nag,resstart,atomstart = rechain_renumber(cag,'A',0,0)
    nab,_,_ = rechain_renumber(cab,'B',resstart,atomstart)

    # rebuild the pdb in order
    new_content = nag + nab

    # return lines or write file
    if output_path is not None:
        write_model(new_content,output_path)
        return output_path

    return new_content




if __name__ == '__main__':

    pdbfile = sys.argv[1]
    chains_AG = sys.argv[2].split(',') # chains in the original file
    chains_AB = sys.argv[3].split(',')
    output_path = sys.argv[4] # do we return lines or write a file if output_path is not None
    
    if output_path == 'None' or (not output_path.endswith('.pdb')):
        output_path = None
    
    # usage: pdbfilepath A H,L Yes

    new_content = organizer(pdbfile,chains_AG,chains_AB,output_path)
    print(np.array(new_content))
