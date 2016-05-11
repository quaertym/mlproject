include("cache.jl")
using Knet, ArgParse, JLD

@knet function embed(x; d=0)
  return wdot(x; out=d)
end

@knet function convolve(x; filterCount=0, filterWidth=0, filterHeight=0, o...)
  w = par(; o..., init=Xavier(), dims=(filterHeight, filterWidth, 1, filterCount))
  return conv(w,x; o...)
end

@knet function rnnlayer(x; dropout=0, hidden=0, o...)
  y = lstm(x; out=hidden)
  return drop(y; pdrop=dropout)
end

# For my problem; nlayer = 2, hidden = 300
@knet function multi_lmmodel(x; d=0, batchsize=1, wordLength=wordLength, filterCount=0, filterWidth=0, poolWidth=poolWidth, nlayer=0, hidden=0, vocabsize=0, o...)
  embedding = embed(x; d=d)
  x1 = reshape(embedding; outdims=(d,wordLength,1,batchsize))
  x2 = convolve(x1; filterCount=filterCount, filterWidth=filterWidth, filterHeight=d)
  x3 = reshape(x2; outdims=(poolWidth*filterCount,1,1,batchsize))
  x4 = tanh(x3)
  x5 = pool(x4; window=poolWidth)
  x6 = reshape(x5; outdims=(filterCount,batchsize))
  z = repeat(x6; o..., frepeat=:rnnlayer, nrepeat=nlayer, hidden=hidden)
  return wbf(z; out=vocabsize, f=:soft)
end

function main(args=ARGS)
  println("Let's start...")

  train_file = "data/ptb.train.txt"

  global cache = Cache(train_file)

  # Store dictionary lengths
  charCount = length(cache.char_dict)
  wordCount = length(cache.word_dict)

  # print char dictionary
  println("Character dictionary: $(cache.char_dict)")

  longest_word = cache.longest_word
  longest_length = length(longest_word)
  println("Longest: $longest_word, size: $longest_length")

  d = 15 # Size for embeddings, this is also convolution filter height
  wordLength = longest_length # length(word)
  filterCount = 525 # 12 # Convolution filter count
  filterWidth = 6 # 5 # Colvolution filter width
  poolWidth = (wordLength - filterWidth + 1) # pooling width
  nlayer = 2
  hidden = 300
  
  lr = 1 # 0.1
  numberOfEpochs = 25
  batchsize = 20 # 125
  seqlength = 35 # 100
  dropout = 0

  net = nothing
  try
    net = load("./net.jld", "net")
  catch
    net = compile(:multi_lmmodel; d=d, batchsize=batchsize, filterCount=filterCount, filterWidth=filterWidth, wordLength=wordLength, poolWidth=poolWidth, nlayer=nlayer, hidden=hidden, vocabsize=wordCount)
  end

  currentPerplexity = check_model(net; seqlength=seqlength, batchsize=batchsize)
  @printf("Initial perplexity:%g\n", currentPerplexity)
  flush(STDOUT)

  words = Utils.get_words(train_file)
  multitrain(net, words, currentPerplexity; lr=lr, seqlength=seqlength, batchsize=batchsize, numberOfEpochs=numberOfEpochs)
end

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

function check_model(net; seqlength=35, batchsize=20)
  test_file = "data/ptb.test.txt"
  words = Utils.get_words(test_file)
  return multitest(net, words; seqlength=seqlength, batchsize=batchsize)
end

function multitrain(net, words, currentPerplexity; lr=0.1, seqlength=100, batchsize=125, numberOfEpochs=1)
  batchcount = div(length(words), batchsize * seqlength)
  data = words[1:batchsize*seqlength*batchcount]
  dataLength = length(data) # println(dataLength)

  reshapedData = reshape(data,seqlength,batchsize,batchcount)
  
  setp(net, lr=lr)
  for epoch=1:numberOfEpochs
    sumloss = numloss = 0
    for i=1:batchcount
      batch = get_batch(reshapedData, i)
      sumloss += train(net, batch, seqlength, softloss)
      numloss += 1
      # @printf("completed epoch:%d batch:%d\n", epoch, i)
    end
    @printf("epoch:%d softloss:%g\n", epoch, sumloss/numloss)
    flush(STDOUT)
    perplexity = check_model(net; seqlength=seqlength, batchsize=batchsize)
    if perplexity < currentPerplexity
      @printf("Perplexity improved from %g to %g\n", currentPerplexity, perplexity)
      flush(STDOUT)
      currentPerplexity = perplexity
      save("./net.jld", "net", net)
    end
  end
end

function train(net, batch, seqlength, loss; gclip=5, dropout=0, o...)
  ystack = cell(0)
  sumloss = 0.0
  reset!(net, keepstate=false)
  T = seqlength-1
  for t=1:T
    current_words = get_seq_words(batch, t)
    next_words = get_seq_words(batch, t+1)
    x = convert_to_input(current_words)
    ygold = convert_to_output(next_words)
    ypred = sforw(net,x; dropout=(dropout>0))
    sumloss += loss(ypred, ygold)
    push!(ystack,ygold)
  end
  while !isempty(ystack)
    ygold = pop!(ystack)
    sback(net,ygold,loss)
  end
  g = (gclip > 0 ? gnorm(net) : 0)
  update!(net; gscale=(g > gclip > 0 ? gclip/g : 1))
  reset!(net, keepstate=true)
  return sumloss/T
end

function multitest(net, words; seqlength=100, batchsize=125)
  tokenCount = length(words)
  batchcount = div(tokenCount, batchsize * seqlength)
  data = words[1:batchsize*seqlength*batchcount]
  dataLength = length(data)

  reshapedData = reshape(data,seqlength,batchsize,batchcount)
  
  sumloss = numloss = 0
  for i=1:batchcount
    batch = get_batch(reshapedData, i)
    sumloss += test(net, batch, seqlength, softloss)
    numloss += 1
  end

  # Perplexity
  perplexity = exp(sumloss * batchsize / tokenCount)
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
  sumloss
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
