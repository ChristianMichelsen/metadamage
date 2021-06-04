from pathlib import Path

from ete3 import NCBITaxa
from joblib import Memory


cachedir = "memoization"
memory = Memory(cachedir, verbose=0)

# @lru_cache
@memory.cache
def extract_descendant_tax_ids(tax, include_subspecies=True):
    """Given either tax_id or tax_name, extract all the descendants' taxIDs.
    Subspecies are automatically included, but can be disables with
    include_subspecies = False.
    """

    ncbi = NCBITaxa()

    # species
    try:
        descendants = set(ncbi.get_descendant_taxa(tax, collapse_subspecies=True))
    except ValueError:
        return []

    # subspecies
    if include_subspecies:
        try:
            descendants |= set(ncbi.get_descendant_taxa(tax, collapse_subspecies=False))
        except ValueError:
            pass

    if Path("taxdump.tar.gz").exists():
        Path("taxdump.tar.gz").unlink()

    return list(descendants)


#%%


def is_ipython():
    return hasattr(__builtins__, "__IPYTHON__")


if is_ipython():

    ncbi = NCBITaxa()

    taxid2name = ncbi.get_taxid_translator([131567])
    print(taxid2name)

    name2taxid = ncbi.get_name_translator(
        ["Homo sapiens", "primates", "Homo", "Salmonidae"]
    )
    print(name2taxid)

#%%

# name = "Homo"
# name = 9605
# name = "Ursus"
# name = "Struthio"

# descendants = list(ncbi.get_descendant_taxa(name))
# print("")
# print(ncbi.translate_to_names(descendants))

# # you can easily ignore subspecies, so only taxa labeled as "species" will be reported:
# descendants = list(ncbi.get_descendant_taxa(name, collapse_subspecies=True))
# print("")
# print(ncbi.translate_to_names(descendants))

# # or even returned as an annotated tree
# tree = ncbi.get_descendant_taxa(name, collapse_subspecies=True, return_tree=True)
# print("")
# print(tree.get_ascii(attributes=["sci_name", "tax_id"]))

# # %%
