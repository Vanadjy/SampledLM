function update_sample!(nls, k)
    if (length(nls.data_mem) / nls.nls_meta.nequ) + nls.sample_rate ≤ 1.0 #case where we don't have any data recovery
        #display("no recovery : $((length(nls.data_mem) / nls.nls_meta.nequ) + nls.sample_rate)")
        # creates a new sample which can select indexes which are not yet in data_mem
        nls.sample = sort(shuffle!(setdiff(collect(1:nls.nls_meta.nequ), nls.data_mem))[1:length(nls.sample)])

        #adding to data_mem the indexes contained in the current sample
        nls.data_mem = vcat(nls.data_mem, nls.sample)

    else #case where we have data recovery
        #display("recovery : $((length(nls.data_mem) / nls.nls_meta.nequ) + nls.sample_rate)")
        sample_size = Int(ceil(nls.sample_rate * nls.nls_meta.nequ))
        sample_complete = shuffle!(nls.data_mem)[1:(sample_size + length(nls.data_mem) - nls.nls_meta.nequ)]
        #picks up all the unvisited data and add a random part from the current memory
        nls.sample = sort(vcat(setdiff!(collect(1:nls.nls_meta.nequ), nls.data_mem), sample_complete))

        # adding in memory the sampled data used to complete the sample
        nls.data_mem = sample_complete
        push!(nls.epoch_counter, k)
    end
end

function uniform_sample(length, sample_rate)
    sample = []
    counter = 0.0
    for i in 2:length
        counter += sample_rate
        if counter ≥ 1.0
            push!(sample, i)
            counter -= 1.0
        end
    end
    sample
end

#=function basic_change_sample_rate(epoch_count::Int)
    if (epoch_count >= 0) && (epoch_count <= 1)
        return .05
      elseif (epoch_count > 1) && (epoch_count <= 2)
        return .3
      elseif (epoch_count > 2) && (epoch_count <= 4)
        return .7
      elseif (epoch_count > 4)
        return 1.0
      end
end=#

#=function basic_warn_sample_update(epoch_count::Int)
    if epoch_count ∈ [6, 11, 16]
        return true
    end
end=#

function get_filename(name::AbstractString)
    if name[(end - 2):end] == "bz2"
      filename = name
    elseif name[(end - 2):end] == "txt"
      filename = name * ".bz2"
    elseif name[(end - 2):end] == "pre"
      filename = name * ".txt.bz2"
    elseif occursin(r"[0-9]{3}", name[(end - 2):end])
      filename = name * "-pre.txt.bz2"
    else
      error("Cannot recognize $(name)")
    end
  
    return filename
  end

function sp_sample(rows::AbstractVector{T}, sample::AbstractVector{<:Integer}) where {T}
  sp_sample = Int[]
  for i in eachindex(rows)
    if rows[i] in sample
      push!(sp_sample, i)
    end
  end
  return sp_sample
end

function formatting_tickvals_10power(v)
  tickvals = Vector{Int}(undef, length(v))
  for i in eachindex(tickvals)
    if i <= 10
      tickvals[i] = i
    elseif isinteger(i / 10)
      tickvals[i] = i
    end
  end
  return tickvals
end

function formatting_tickvals_10power(v)
  pmin = floor(log10(minimum(v)))
  pmax = ceil(log10(maximum(v)))
  tickvals = Float64[]
  for i in pmin:(pmax-1)
    if i < 0
      els = [k / 10.0^(-i) for k in 1.0:9.0]
      append!(tickvals, els)
    else
      els = [k * 10.0^i for k in 1.0:9.0]
      append!(tickvals, els)
    end
  end
  tickvals = vcat(tickvals, 10^pmax)
  return tickvals
end

function log_scale(n)
  try
    Int(log10(n))
  catch
    error("Input Error: n should be a power of 10")
  end
  log_scale = [k * 10.0^(i) for i in 0:Int(log10(n) - 1) for k in 1.0:9.0]
  return Int.(log_scale)
end