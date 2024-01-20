function uniform_sample(length, sample_rate)
    sample = [1]
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

function my_zero(x)
    return 0.0
end