function embedded_matrix = takens_embedding(time_series, embedding_dimension, time_delay)
    % Create the embedded phase space matrix
    N = length(time_series);
    embedded_matrix = zeros(N - (embedding_dimension - 1) * time_delay, embedding_dimension);
    for i = 1:embedding_dimension
        embedded_matrix(:, i) = time_series((i - 1) * time_delay + 1 : end - (embedding_dimension - i) * time_delay);
    end
end