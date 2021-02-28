### Subset Generator

`generator_subset.py` - run to generate a subset based on the initial dataset from the paper.

Picks an arbitrary point in KG and goes `DEPTH` into all directions to copy over.

- in the file header edit the constants:
  - `DEPTH` to edit the size of the resulting dataset. Growth is exponential.
  - path-related constants:
    - `path_base` - directory that presumably contains all the files required
    - other files like `ent_links` - path relative of `path-base`. In ordinary cases shouldnt be edited. If the
    dir layout differs from the expected - adopt the vaulues for specified files accordingly.

### Args

file `args.json` needs to be edited as well. Learning rates influence the results.
