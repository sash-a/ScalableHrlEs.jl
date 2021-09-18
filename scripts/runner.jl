include("runutils.jl")
args = parseargs()
run(ScalableHrlEs.loadconfig(args["cfgpath"]), args["mjpath"])