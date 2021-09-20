using Configurations

@option struct Env
    name::String
    steps::Int
end

@option struct Training
    episodes::Int
    policies::Int
    generations::Int
end

@option struct Hrl
    pretrained::String
    interval::Int
    cdist::Float32
    onehot::Bool
end

@option struct SHrlEsConfig
    name::String
    env::Env
    training::Training
    hrl::Hrl
end

loadconfig(file::String) = from_dict(SHrlEsConfig, YAML.load_file(file; dicttype=Dict{String, Any}))