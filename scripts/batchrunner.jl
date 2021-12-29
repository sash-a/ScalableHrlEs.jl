include("runutils.jl")

using YAML
using Pidfile

# merging nested dicts
recursive_merge(x::AbstractDict...) = merge(recursive_merge, x...)
recursive_merge(x::AbstractVector...) = cat(x...; dims=1)
recursive_merge(x...) = x[end]

function getconfig(batchfilepath)
    cfgpath = ""
    nrun = -1
    name = ""
    overrides = Dict{String, Any}()

    batchconfig = YAML.load_file(batchfilepath; dicttype=Dict{String, Any})
    for (_, options) in batchconfig
        if options["nruns"] > 0
            options["nruns"] -= 1

            nrun = options["nruns"]

            name = haskey(options, "name") ? options["name"] : name
            overrides = haskey(options, "overrides") ? options["overrides"] : overrides
            cfgpath = options["path"]

            YAML.write_file(batchfilepath, batchconfig)
            break
        end
    end

    if nrun != -1 && !isempty(cfgpath)
        cfg_dict = YAML.load_file(cfgpath; dicttype=Dict{String, Any})
        merged_dict = recursive_merge(cfg_dict, overrides)
        conf = ScalableHrlEs.loadconfig(merged_dict)
        conf = ScalableHrlEs.SHrlEsConfig("$(conf.name)_$(name)_$nrun", conf.seed, conf.env, conf.training, conf.hrl)
        conf
    else
        nothing
    end
end

function batchrun()
    args = parseargs()

    comm = if args["mpi"]
        MPI.Init()
        MPI.COMM_WORLD  # expecting this to be one per node
    else
        ScalableES.ThreadComm()
    end

    batchfilepath = args["cfgpath"]
    
    conf = nothing
    if ScalableES.isroot(comm)
        conf = mkpidlock("ses-batchfile-lock") do
            @show getconfig(batchfilepath)
        end
    end
    
    if ScalableES.ismpi(comm)
        conf = MPI.bcast(conf, ScalableES.MPIROOT, comm)
    end
    
    @show conf
    @assert conf !== nothing

    run(conf, args["mjpath"], comm)
end

batchrun()