# List of elements the most comon in nature and used for formula encoding
ELEMENTS = ['C', 'H', 'N', 'O', 'S', 'P', 'Cl', 'Br', 'F', 'I']

# """Global variables used across the package."""
# import pathlib
#
# # Dirs
# PHANTOMS_ROOT_DIR = pathlib.Path(__file__).parent.absolute()
# PHANTOMS_REPO_DIR = PHANTOMS_ROOT_DIR.parent
# PHANTOMS_DATA_DIR = PHANTOMS_REPO_DIR / 'data'
# PHANTOMS_TEST_RESULTS_DIR = PHANTOMS_DATA_DIR / 'test_results'
# PHANTOMS_ASSETS_DIR = PHANTOMS_REPO_DIR / 'assets'

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# Chemistry
# List of all 118 elements (indexed by atomic number)
CHEM_ELEMS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]