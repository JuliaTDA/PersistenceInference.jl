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
        "Point-cloud inference" => "point_clouds.md",
        "Functional confidence bands" => "functional_inference.md",
        "Null maxima and independence" => "hypothesis_tests.md",
        "Practical choices and validation" => "practical.md",
        "API reference" => "api.md",
    ],
)

if get(ENV, "JULIATDA_DOCS_DEPLOY", "false") == "true"
    deploydocs(;
        repo = "github.com/JuliaTDA/PersistenceInference.jl",
        devbranch = "main",
    )
end
