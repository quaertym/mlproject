module Utils
  function create_char_dict_from_file(filename)
    vocab = Dict{Char,Int}()
    text = readall(filename)
    for char in text; get!(vocab, char, 1+length(vocab)); end
    vocab_size = length(vocab)
    info("$(vocab_size) unique chars")
    return vocab
  end

  function create_word_dict_from_file(filename)
    vocab = Dict{AbstractString,Int}()
    lines = readlines(filename)
    for line in lines;
      for word in split(line);
        get!(vocab, word, 1+length(vocab));
      end
    end
    vocab_size = length(vocab)
    info("$(vocab_size) unique words")
    return vocab
  end

  function build_dicts_from_words(words)
    # Dictionary to hold uniq characters and words
    char_dict = Dict{Char,Int}()
    word_dict = Dict{AbstractString,Int}()

    # For each line read and assign index for each word and character
    for word in words
      get!(word_dict, word, 1+length(word_dict));
      for char in word
        get!(char_dict, char, 1+length(char_dict));
      end
    end

    # Print dictionary lengths
    word_dict_size = length(word_dict)
    char_dict_size = length(char_dict)

    info("$(length(words)) tokens")
    info("$(word_dict_size) unique words")
    info("$(char_dict_size) unique chars")

    # Return both dictionaries as tuple
    return (word_dict, char_dict)
  end

  function build_dicts(filename)
    words = get_words(filename)
    return build_dicts_from_words(words)
  end

  function get_words(filename)
    text = readall(filename)
    replaced = replace(text, "\n", "<eos>")
    words = split(replaced)
    info("File \"$(filename)\" contains $(length(words)) words.")
    return words
  end

  function create_one_hot_matrix(word, char_dict, longestLength)
    # v = zeros(Float32, length(char_dict), length(word))
    v = zeros(Float32, length(char_dict), longestLength)
    for (index, char) in enumerate(word); v[char_dict[char], index] = 1 end
    return sparse(v)
  end

  function create_one_hot_vector(char, char_dict)
    v = zeros(Float32, length(char_dict),1)
    v[char_dict[char]] = 1
    return sparse(v)
  end

  function get_longest_word(word_dict)
    longest_word = nothing
    for word in keys(word_dict)
      if longest_word == nothing || length(word) > length(longest_word)
        longest_word = word
      end
    end
    return longest_word
  end
end