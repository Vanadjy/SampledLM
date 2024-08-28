@testset "IJCNN1-Train" begin
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
    nlp_train, nls_train, sol = ijcnn1_train_model()

    test_well_defined(nlp_train, nls_train, sol)
    test_objectives(nlp_train, nls_train)
  
    @test nlp_train.meta.nvar == 22
    @test nls_train.nls_meta.nequ == 49990
    @test all(nlp_train.meta.x0 .== 1)
    @test length(findall(x -> x .!= -1, sol)) == 4853
    @test length(findall(x -> x .!= 1, sol)) == 45137
end

@testset "IJCNN1-Test" begin
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
    nlp_train, nls_train, sol = ijcnn1_test_model()

    test_well_defined(nlp_train, nls_train, sol)
    test_objectives(nlp_train, nls_train)
  
    @test nlp_train.meta.nvar == 22
    @test nls_train.nls_meta.nequ == 91701
    @test all(nlp_train.meta.x0 .== 1)
    @test length(findall(x -> x .!= -1, sol)) == 8712
    @test length(findall(x -> x .!= 1, sol)) == 82989
end