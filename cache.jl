include("utils.jl")

type Cache
  inputCache
  outputCache
  word_dict
  char_dict
  longest_word
  longest_word_length
  word_dict_size
  char_dict_size

  # Cache(word_dict, char_dict) = new(Dict{AbstractString,Any}(), Dict{AbstractString,Any}(), word_dict, char_dict, 0)
  Cache() = new(nothing, nothing, nothing, nothing, nothing, 0, 0, 0)

  function call(::Type{Cache}, file::AbstractString)
    c = Cache()

    dicts = Utils.build_dicts(file)
    word_dict = dicts[1]
    char_dict = dicts[2]

    c.inputCache = Dict{AbstractString,Any}()
    c.outputCache = Dict{AbstractString,Any}()
    c.word_dict = word_dict
    c.char_dict = char_dict
    c.longest_word = Utils.get_longest_word(c.word_dict)
    c.longest_word_length = length(c.longest_word)
    c.word_dict_size = length(word_dict)
    c.char_dict_size = length(char_dict)

    return c
  end
end

function build(c::Cache, word_dict, char_dict)
  c.inputCache = Dict{AbstractString,Any}()
  c.outputCache = Dict{AbstractString,Any}()
  c.word_dict = word_dict
  c.char_dict = char_dict
  c.longest_word = Utils.get_longest_word(c.word_dict)
  c.longest_word_length = length(c.longest_word)

  # Populate cache, uncomment this for eager loading
  # for word in keys(c.word_dict)
  #   get_input(c, word)
  #   get_output(c, word)
  # end
end

function build_from_file(c::Cache, file)
  dicts = Utils.build_dicts(file)
  word_dict = dicts[1]
  char_dict = dicts[2]
  build(c, word_dict, char_dict)
end

function get_input(c::Cache, word)
  word = (haskey(c.word_dict, word)) ? word : "<unk>"
  cache = c.inputCache
  if !haskey(cache, word)
    cache[word] = Utils.create_one_hot_matrix(word, c.char_dict, c.longest_word_length)
  end
  return cache[word]
end

function get_output(c::Cache, word)
  word = (haskey(c.word_dict, word)) ? word : "<unk>"
  cache = c.outputCache
  if !haskey(cache, word)
    cache[word] = Utils.create_one_hot_vector(word, c.word_dict)
  end
  return cache[word]
end
