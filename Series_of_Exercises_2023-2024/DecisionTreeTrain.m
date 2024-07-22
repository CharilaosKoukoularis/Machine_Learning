%                         Μηχανική Μάθηση
%                   1η Σειρά Ασκήσεων 2023-2024
%                   Άσκηση 1.6: Decision Trees
% -------------------------------------------------------------------------
%                   Χαρίλαος Κουκουλάρης el18137
%                           1/12/2023

% Αλγόριθμος DecisionTree.Train() από τις διαφάνειες του κ.Στάμου
function DecisionTreeTrain(data,criterion)
    
    % 1. Επίλεξε ένα χαρακτηριστικό εισόδου a που παίρνει διαφορετικές τιμές στο 𝔻
    node = FindNextNode(data,criterion);

    if (node <= 0)
        if (node == -1)
            '-1'
        elseif (node == -2)
            '+1'
        else
            '0'
        end
        return
    end
    
    data.Properties.VariableNames{node}
    % 2. Φτιάξε ένα νέο κόμβο A και όρισε το a ως χαρακτηριστικό απόφασης
    % Σε ξεχωριστό σχήμα
    
    % 3. Για κάθε διαφορετική τιμή του a:
    values = unique(data(:,node).Variables);
    for i = 1:1:length(values)
    
        % 3.1 Φτιάξε ένα νέο κόμβο ως παιδί του τρέχοντος κόμβου
        root = data(:,node);
    
        %{ 3.2 Όρισε το 𝔻′ ως το υποσύνολο του 𝔻 με τα στοιχεία που έχουν τη 
        %  τιμή αυτή για το a
        %}
        sample_mask = root.Variables == values(i);
        feature_mask = 1:1:width(data) ~= node;
        new_data = data(sample_mask,feature_mask);
        new_data
    
        %{ 3.3 Αν όλα τα y ∈ 𝔻′ έχουν την ίδια ετικέτα, τότε κάνε τον A φύλλο 
        %  με τιμή την ετικέτα αυτή. 
        %  Αλλιώς 𝖣𝖾𝖼𝗂𝗌𝗂𝗈𝗇𝖳𝗋𝖾𝖾.𝗍𝗋𝖺𝗂𝗇(𝔻′)
        %}
        DecisionTreeTrain(new_data,criterion)
    end
end