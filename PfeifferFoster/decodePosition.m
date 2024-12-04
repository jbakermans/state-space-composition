function decoded = decodePosition(sum_ratemaps, log_ratemaps, counts, t)
    % Probability of positions x given spikes n:
    % p(x|n) = P / sum_j-M P, where j counts M position bins
    % P = [prod_i-N f_i(x)^n_i] exp(-t sum_i-N f_i(x))
    % where i counts N neurons, f_i is ratemap of neuron i, n_i is spikes of neuron i, t is time window
    % Also see https://www.cell.com/cms/10.1016/j.neuron.2009.07.027/attachment/f0e8c7f0-815b-46eb-bbdd-8ebcd095568e/mmc1.pdf
    
    % I'm going to do this in log space to avoid numerical errors
    % log(P) = sum_i-N n_i * log(f_i(x)) - t * sum_i-N f_i(x)
    % But it does introduce a problem: 0^0 = 1, but for log that becomes 0*-inf = Nan,
    % so replace any inf by some arbitrary large number
    % Finally, t sum_i-N f_i(x) is always the same so precalculate it

    % This is a super ugly but vectorised way of calculating n_i * log(f_i(x))    
    n_log_ratemaps = reshape(...
        repmat(counts, [1, size(log_ratemaps,2) * size(log_ratemaps,3)]) .* ... % repeat n_i for each ratemap to get neurons x (space * space)
        reshape(log_ratemaps, [size(log_ratemaps,1), size(log_ratemaps,2)*size(log_ratemaps,3)]) ... % turn ratemaps into flat neurons x (space * space) vectors
        , [size(log_ratemaps,1), size(log_ratemaps, 2), size(log_ratemaps, 3)]); % turn ratemap vectors back into neurons x space x space matrices
    log_P = squeeze(sum(n_log_ratemaps,1) - t * sum_ratemaps);
    % For normalisation, I can add constant to each log before exponential
    decoded = exp(log_P);
    decoded = decoded ./ sum(decoded(~isnan(decoded)));
end