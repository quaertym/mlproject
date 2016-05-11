using Knet, ArgParse
include("cache.jl")

function main(args=ARGS)
    global net, vocab, text, data
    s = ArgParseSettings()
    s.description="Print stats for different files"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--file"; help="data file to process")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)

    Cache(o[:file])
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)