#TODO : add the necessary fields for SampledNLSModel to make the following function work

function update_sample!(nls::SampledNLSModel)
    nls.sampler = sort(randperm(nls.nls_meta.nequ)[1:length(nls.sampler)])
    @inbounds for index in nls.sampler
        if (nls.len_mem < nls.nls_meta.nequ) && (index ∉ nls.sampler_mem)
          push!(nls.sampler_mem, index) # keeps in memory all the visited indexes
          nls.len_mem += 1 # indicates how many indexes have been visited
        end
    end
    #sampler, sampler_mem, len_mem
  end