include("runutils.jl")

using YAML
using Pidfile


function getconfig(batchfilepath)
    cfgpath = ""
    nrun = -1
    batchconfig = YAML.load_file(batchfilepath; dicttype=Dict{String, Any})
    for (path, options) in batchconfig
        if options["nruns"] > 0
            options["nruns"] -= 1

            nrun = options["nruns"]
            cfgpath = path

            YAML.write_file(batchfilepath, batchconfig)
            break
        end
    end

    if nrun != -1 && !isempty(cfgpath)
        conf = ScalableHrlEs.loadconfig(cfgpath)
        conf = ScalableHrlEs.SHrlEsConfig("$(conf.name)_$nrun", conf.env, conf.training, conf.hrl)
        conf
    else
        nothing
    end
end

function batchrun()
    args = parseargs()
    batchfilepath = args["cfgpath"]
    
    batchfilepath = "config/batch.yml"

    conf = mkpidlock("ses-batchfile-lock") do
        @show getconfig(batchfilepath)
    end
    @show conf
    @assert conf !== nothing

    run(conf, args["mjpath"])

end

batchrun()