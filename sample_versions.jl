# Version 1: List of predetermined - switch with mobile average #
if version == 1
    # Change sample rate
    #nls.sample_rate = basic_change_sample_rate(epoch_count)
    if nls.sample_rate < sample_rates_collec[end]
        Num_mean = Int(ceil(1 / nls.sample_rate))
        if k >= Num_mean
        @views mobile_mean = mean(Fobj_hist[(k - Num_mean + 1):k] + Hobj_hist[(k - Num_mean + 1):k])
        if abs(mobile_mean - (fk + hk)) ≤ 1e-1 #if the mean on the Num_mean last iterations is near the current objective value
            nls.sample_rate = sample_rates_collec[sample_counter]
            sample_counter += 1
            change_sample_rate = true
        end
        end
    end
end

# Version 2: List of predetermined - switch with arbitrary epochs #
if version == 2
    if nls.sample_rate < 1.0
        if epoch_count > epoch_limits[sample_counter]
        nls.sample_rate = sample_rates_collec[sample_counter]
        sample_counter += 1
        change_sample_rate = true
        end
    end
end

# Version 3: Adapt sample_size after each iteration #
if version == 3
    # ζk = Int(ceil(k / (1e8 * min(1, 1 / μk^4))))
    p = .75
    q = .75
    ζk = Int(ceil(100 * (log(1 / (1-p)) * max(μk^4, μk^2) + log(1 / (1-q)) * μk^4)))
    nls.sample_rate = min(1.0, (ζk / nls.nls_meta.nequ) * (nls.meta.nvar + 1))
    change_sample_rate = true
end

# Version 4: Double sample_size after a fixed number of epochs or a mobile mean stagnation #
if version == 4
    # Change sample rate
    #nls.sample_rate = basic_change_sample_rate(epoch_count)
    if nls.sample_rate < 1.0
        Num_mean = Int(ceil(1 / nls.sample_rate))
        if k >= Num_mean
            @views mobile_mean = mean(Fobj_hist[(k - Num_mean + 1):k] + Hobj_hist[(k - Num_mean + 1):k])
            if abs(mobile_mean - (fk + hk)) ≤ 1e-1 #if the mean on the Num_mean last iterations is near the current objective value
                nls.sample_rate = min(1.0, 2 * nls.sample_rate)
                change_sample_rate = true
                unchange_mm_count = 0
            else # don't have stagnation
                unchange_mm_count += nls.sample_rate
                if unchange_mm_count ≥ 3 # force to change sample rate after 3 epochs of unchanged sample rate using mobile mean criterion
                nls.sample_rate = min(1.0, 2 * nls.sample_rate)
                change_sample_rate = true
                unchange_mm_count = 0
                end
            end
        end
    end
end

# Version 5: change sample rate when gain factor 10 accuracy #
if version == 5
    if k == 1
        ξ_mem = Metric_hist[1]
    end
    if nls.sample_rate < sample_rates_collec[end]
        #@views mobile_mean = mean(Fobj_hist[(k - Num_mean + 1):k] + Hobj_hist[(k - Num_mean + 1):k])
        if metric/ξ_mem ≤ 1e-1 #if the current metric is a factor 10 lower than the previously stored ξ_mem
        nls.sample_rate = sample_rates_collec[sample_counter]
        sample_counter += 1
        ξ_mem *= 1e-1
        change_sample_rate = true
        end
    end
end

# Version 6: Double sample_size after a fixed number of epochs or a metric decrease #
if version == 6
    if k == 1
        ξ_mem = Metric_hist[1]
    end
    # Change sample rate
    #nls.sample_rate = basic_change_sample_rate(epoch_count)
    if nls.sample_rate < 1.0
        if metric/ξ_mem ≤ 1e-1 #if the mean on the Num_mean last iterations is near the current objective value
        nls.sample_rate = sample_rates_collec[sample_counter]
        sample_counter += 1
        ξ_mem *= 1e-1
        change_sample_rate = true
        unchange_mm_count = 0
        else # don't get more accurate ξ
            unchange_mm_count += nls.sample_rate
            if unchange_mm_count ≥ 3 # force to change sample rate after 3 epochs of unchanged sample rate using mobile mean criterion
                nls.sample_rate = sample_rates_collec[sample_counter]
                sample_counter += 1
                change_sample_rate = true
                unchange_mm_count = 0
            end
        end
    end
end

if version == 7
    if (count_fail == 2) && nls.sample_rate != sample_rate0 # if μk increased 3 times in a row -> decrease the batch size AND useless to try to make nls.sample rate decrease if its already equal to sample_rate0
        sample_counter = max(0, sample_counter - 1) # sample_counter-1 < length(sample_rates_collec)
        nls.sample_rate = (sample_counter == 0) ? sample_rate0 : sample_rates_collec[sample_counter]
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
    elseif (count_big_succ == 2) && nls.sample_rate != sample_rates_collec[end] # if μk decreased 3 times in a row -> increase the batch size AND useless to try to make nls.sample rate increase if its already equal to the highest available sample rate
        sample_counter = min(length(sample_rates_collec), sample_counter + 1) # sample_counter + 1 > 0
        nls.sample_rate = sample_rates_collec[sample_counter]
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
    end
end

if version == 8
    if (count_fail == 3) && nls.sample_rate != sample_rate0 # if μk increased 3 times in a row -> decrease the batch size AND useless to try to make nls.sample rate decrease if its already equal to sample_rate0
        nls.sample_rate -= δ_sample
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
    elseif (count_big_succ == 3) && nls.sample_rate != sample_rates_collec[end] # if μk decreased 3 times in a row -> increase the batch size AND useless to try to make nls.sample rate increase if its already equal to the highest available sample rate
        nls.sample_rate += δ_sample
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
    end
end

if (version == 9)
    if nls.sample_rate < 1.0
        if (count_fail == 2) && nls.sample_rate != sample_rates_collec[end] # if μk increased twice in a row -> decrease the batch size AND useless to try to make nls.sample rate decrease if its already equal to sample_rate0
        #=ζk *= λ^4
        @info "possible sample rate = $((ζk / nls.nls_meta.nequ) * (nls.meta.nvar + 1))"
        nls.sample_rate = min(1.0, max((ζk / nls.nls_meta.nequ) * (nls.meta.nvar + 1), buffer))=#
        nls.sample_rate = min(1.0, max(nls.sample_rate * λ, buffer))
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
        count_succ = 0
        dist_succ = zero(eltype(xk))
        elseif (count_big_succ == 2) && nls.sample_rate != sample_rate0 # if μk decreased twice in a row -> increase the batch size AND useless to try to make nls.sample rate increase if its already equal to the highest available sample rate
        #ζk *= λ^(-4)
        #@info "possible sample rate = $((ζk / nls.nls_meta.nequ) * (nls.meta.nvar + 1))"
        nls.sample_rate = min(1.0, max(nls.sample_rate / λ, buffer))
        change_sample_rate = true
        count_fail = 0
        count_big_succ = 0
        count_succ = 0
        dist_succ = zero(eltype(xk))
        end
        if (nls.sample_rate < sample_rates_collec[end]) && ((dist_succ > (norm(ones(nls.meta.nvar)) / (threshold_relax * nls.sample_rate))) || (count_succ > 10)) # if μ did not change for too long, increase the buffer value
        @info "sample rate buffered at $(sample_rates_collec[sample_counter] * 100)%"
        buffer = sample_rates_collec[sample_counter]
        nls.sample_rate = min(1.0, max(nls.sample_rate, buffer))
        sample_counter += 1
        change_sample_rate = true
        count_succ = 0
        dist_succ = zero(eltype(xk))
        end
    end
end