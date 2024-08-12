export ijcnn1_generate_data_tr, ijcnn1_generate_data_test, ijcnn1_model_sto

#packages for getting and decompressing IJCNN1 from LIBSVM
using HTTP, CodecBzip2

function ijcnn1_generate_data_tr(; n::Int = 49990, d::Int = 22)
    #getting data
    data = HTTP.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2")
    bz2_file = transcode(Bzip2Decompressor, data.body)
    data_file = String(bz2_file)
    lines = split(data_file, '\n')

    #allocate memory to store data
    A = zeros(Float64, n, d)
    y = zeros(Float64, n)
    i = 1
    while i ≤ n
        dummy = split(lines[i], ' ')
        y[i] = parse(Float64, dummy[1])
        for j in 2:(length(dummy))
            loc_val = split(dummy[j],':')
            if length(loc_val) < 2
                display(i)
                display(loc_val)
            end
            A[i, parse(Int, loc_val[1])] = parse(Float64, loc_val[2])
        end
    i += 1
    end
    A, y
end

function ijcnn1_generate_data_test(; n::Int = 91701, d::Int = 22)
    #getting data
    data = HTTP.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2")
    bz2_file = transcode(Bzip2Decompressor, data.body)
    data_file = String(bz2_file)
    lines = split(data_file, '\n')

    #allocate memory to store data
    A = zeros(Float64, n, d)
    y = zeros(Float64, n)
    i = 1
    while i ≤ n
        dummy = split(lines[i], ' ')
        y[i] = parse(Float64, dummy[1])
        for j in 2:(length(dummy))
            loc_val = split(dummy[j],':')
            if length(loc_val) < 2
                display(i)
                display(loc_val)
            end
            A[i, parse(Int, loc_val[1])] = parse(Float64, loc_val[2])
        end
    i += 1
    end
    A, y
end

function ijcnn1_train_model()
    A_tr, b_tr = ijcnn1_generate_data_tr()
    svm_model(A_tr', b_tr)
end

function ijcnn1_test_model()
    A_test, b_test = ijcnn1_generate_data_test()
    svm_model(A_test', b_test)
end