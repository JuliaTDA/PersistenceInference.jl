using Documenter
using PersistenceInference

DocMeta.setdocmeta!(PersistenceInference, :DocTestSetup,
                    :(using PersistenceInference); recursive = true)

makedocs(;
    modules = [PersistenceInference],
    sitename = "PersistenceInference.jl",
    authors = "G. Vituri and contributors",
    # The docstrings' worked examples use random data and recompute persistent
    # homology, so they are illustrative rather than doctests; skip doctesting.
    doctest = false,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaTDA.github.io/PersistenceInference.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/JuliaTDA/PersistenceInference.jl",
    devbranch = "main",
)
