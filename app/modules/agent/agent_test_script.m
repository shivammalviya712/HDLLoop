function result = nested_loops_example(A, B)
    result = zeros(size(A));
    for i = 1:length(A)
        for j = 1:length(B)
            result(i) = result(i) + A(i) * B(j);
        end
    end
end
