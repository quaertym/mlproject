include("cache.jl")
using Knet, JLD

# This takes reshaped word data and
# returns batch for given index
function get_batch(data, index)
  return data[:,:,index]
end

# This takes batch and returns word at
# index for all sequences in the batch
function get_seq_words(batch, wordIndex)
  return batch[wordIndex, :]
end

# This takes word array and converts them to
# input matrices
function convert_to_input(words)
  a=Any[]
  for i=1:length(words)
    push!(a, get_input(cache, words[i]))
  end
  return hcat(a...)
end

# This takes word array and converts them to
# output vector
function convert_to_output(words)
  a=Any[]
  for i=1:length(words)
    push!(a, get_output(cache, words[i]))
  end
  return hcat(a...)
end

function main(args=ARGS)
  global net = load("./net.jld", "net")

  global cache = Cache("data/ptb.train.txt")
  batchsize = 20
  seqlength = 35

  test_file = "data/ptb.test.txt"
  words = Utils.get_words(test_file)
  multitest(net, words; seqlength=seqlength, batchsize=batchsize)
end

function multitest(net, words; seqlength=100, batchsize=125)
  tokenCount = length(words)
  batchcount = div(tokenCount, batchsize * seqlength)
  println("batch count: $batchcount")
  flush(STDOUT)
  data = words[1:batchsize*seqlength*batchcount]
  dataLength = length(data) # println(dataLength)

  reshapedData = reshape(data,seqlength,batchsize,batchcount)
  
  sumloss = numloss = 0
  for i=1:batchcount
    batch = get_batch(reshapedData, i)
    sumloss += test(net, batch, seqlength, softloss)
    numloss += 1
    @printf("sumloss:%g batch:%d\n", sumloss,i)
    flush(STDOUT)
    # @printf("completed epoch:%d batch:%d\n", epoch, i)
  end
  @printf("end of test\n")

  # Perplexity
  perplexity = exp(sumloss * batchsize / tokenCount)
  # @printf("softloss:%g\n", sumloss/numloss)
  @printf("perplexity: %g, softloss:%g\n", perplexity, sumloss)
  flush(STDOUT)
  return perplexity
end

function test(net, batch, seqlength, loss)
  sumloss = 0.0
  reset!(net)
  T = seqlength-1
  for t=1:T
    current_words = get_seq_words(batch, t)
    next_words = get_seq_words(batch, t+1)
    x = convert_to_input(current_words)
    ygold = convert_to_output(next_words)
    ypred = forw(net,x)
    sumloss += loss(ypred, ygold)
  end
  sumloss # sumloss/T
end

@gpu atexit(()->(for r in net.reg; r.out0!=nothing && Main.CUDArt.free(r.out0); end))

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
