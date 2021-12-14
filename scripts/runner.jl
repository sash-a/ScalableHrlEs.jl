include("runutils.jl")

function main()
    args = parseargs()
    run(ScalableHrlEs.loadconfig(args["cfgpath"]), args["mjpath"], args["mpi"])
end

main()