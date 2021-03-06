using Configurations
import Base.convert

function Base.convert(::Type{Symbol}, s::String)
    @show s
    s = Symbol(s)
    s
end

@option struct Env
    name::String
    steps::Int
    kwargs::Dict{Symbol,Any} = Dict{Symbol,Any}()
end

@option struct Training
    episodes::Int
    policies::Int
    generations::Int

    pop_size::Int = 3
    behv_freq::Int = 25
    min_nov_w::Float32 = 0.8
    sigma::Float32 = 0.02
    lr::Float32 = 0.01
end

@option struct Hrl
    pretrained_ctrl::String = ""
    pretrained_prim::String = ""
    interval::Int
    cdist::Float32
    onehot::Bool
    prim_specific_obs::Bool = false
end

@option struct SHrlEsConfig
    name::String
    seed::Int = 123
    env::Env
    training::Training
    hrl::Hrl
end

function loadconfig(cfg_dict::Dict)
    @show cfg_dict
    # explicit conversion from string to symbol because for some reason overiding Base.convert doesn't work
    if haskey(cfg_dict["env"], "kwargs")
        cfg_dict["env"]["kwargs"] = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(cfg_dict["env"]["kwargs"]))
    end
    from_dict(SHrlEsConfig, cfg_dict)
end

function loadconfig(file::String)
    cfg_dict = YAML.load_file(file; dicttype = Dict{String,Any})
    loadconfig(cfg_dict)
end
