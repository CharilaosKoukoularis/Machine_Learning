%                         Μηχανική Μάθηση
%                   1η Σειρά Ασκήσεων 2023-2024
%                   Άσκηση 1.6: Decision Trees
% -------------------------------------------------------------------------
%                   Χαρίλαος Κουκουλάρης el18137
%                           1/12/2023

function i = FindNextNode(data,criterion)

    if (width(data) <= 1)
        i = 0;
        return
    end

    % Πιθανότητες το y να λάβει τις τιμές +1 και -1
    p = sum(data.y == 1) / height(data);
    n = sum(data.y == -1) / height(data);

    %{
    if (p == 0)
        i = -1;
        return
    end
    if (p == 1)
        i = -2;
        return
    end
    %}

    % Πλήθος μονάδων και μηδενικών σε κάθε χαρακτηριστικό για κάθε έξοδο
    %
    % Συμβολισμός vij: 
    %   κάθε γραμμή αντιστοιχεί και σε ένα χαρακτηριστικό
    %   i: τιμή του χαρακτηριστικού (1 ή 0)
    %   j: τιμή της εξόδου y (p για +1 και n για -1)
    v1p = zeros(width(data)-1,1);
    v1n = zeros(width(data)-1,1);
    v0p = zeros(width(data)-1,1);
    v0n = zeros(width(data)-1,1);

    pmask = data(:,end).Variables == 1;
    nmask = data(:,end).Variables == -1;

    for i = 1:1:width(data)-1
        x = data(:,i).Variables;
        v1p(i) = sum(x(pmask) == 1);
        v1n(i) = sum(x(nmask) == 1);
        v0p(i) = sum(x(pmask) == 0);
        v0n(i) = sum(x(nmask) == 0);
    end
    
    % Συνολικό πλήθος μονάδων και μηδενικών σε κάθε χαρακτηριστικό
    v1 = v1p + v1n;
    v0 = v0p + v0n;
    
    % Πιθανότητες τιμών χαρακτηριστικών να οδηγήσουν σε συγκεκριμένες εξόδους
    p1 = v1p ./ v1;
    n1 = v1n ./ v1;
    p0 = v0p ./ v0;
    n0 = v0n ./ v0;

    p1(isnan(p1)) = 0;
    n1(isnan(n1)) = 0;
    p0(isnan(p0)) = 0;
    n0(isnan(n0)) = 0;
    
    % Εντροπία και Δείκτης gini της ρίζας του δέντρου
    root_entropy = - p * log2(p) - n * log2(n);
    root_gini = 1 - p^2 - n^2;
    
    % 
    text = '';
    for i = 1:1:width(data)-1
        text = append(text,sprintf('[%s]\ty %s=1 %s=0\n\t|- +1: %d\t %d\n\t|- -1: %d\t %d\n', ...
                                        data.Properties.VariableNames{i}, ...
                                        data.Properties.VariableNames{i}, ...
                                        data.Properties.VariableNames{i}, ...
                                        v1p(i), ...
                                        v0p(i), ...
                                        v1n(i), ...
                                        v0n(i)));
    end
    
    % Εντροπία 
    entropy1 = zeros(width(data)-1,1);
    entropy0 = zeros(width(data)-1,1);
    entropy = zeros(width(data)-1,1);
    ig_entropy = zeros(width(data)-1,1);
    for i = 1:1:width(data)-1
        entropy1(i) = - p1(i) * log2(p1(i) + (p1(i) == 0)) - n1(i) * log2(n1(i) + (n1(i) == 0));
        entropy0(i) = - p0(i) * log2(p0(i) + (p0(i) == 0)) - n0(i) * log2(n0(i) + (n0(i) == 0));
        entropy(i) = v1(i) / height(data) * entropy1(i) + v0(i) / height(data) * entropy0(i);
        ig_entropy(i) = root_entropy - entropy(i);
    end
    
    % Δείκτης gini
    gini1 = zeros(width(data)-1,1);
    gini0 = zeros(width(data)-1,1);
    ig_gini = zeros(width(data)-1,1);
    for i = 1:1:width(data)-1
        gini1(i) = 1 - p1(i)^2 - n1(i)^2;
        gini0(i) = 1 - p0(i)^2 - n0(i)^2;
        ig_gini(i) = root_gini - v1(i) / height(data) * gini1(i) - v0(i) / height(data) * gini0(i);
    end
    
    text
    ig_entropy
    ig_gini;
    n1;
    n0;
    p1;
    p0;
    entropy;

    % TODO: move this
    if (p == 0)
        i = -1;
        return
    end
    if (p == 1)
        i = -2;
        return
    end
    
    if (criterion == 'entropy')
        [~,i] = max(ig_entropy);
    else
        [~,i] = max(ig_gini);
    end
end