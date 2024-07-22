%                         Μηχανική Μάθηση
%                   1η Σειρά Ασκήσεων 2023-2024
%          Άσκηση 1.4: Perceptron - MultiLayer Perceptron
% -------------------------------------------------------------------------
%                   Χαρίλαος Κουκουλάρης el18137
%                           28/11/2023


w0 = [1; 0; 0; 0];
x = [[1; 4; 3; 6], [1; 2; -2; 3], [1; 1; 0; -3], [1; 4; 2; 3]];
t = [-1, +1, +1, -1];
class = ["Cn", "Cp", "Cp", "Cn"];
b = 0.1;

%k = 0;
w = w0;
text = '';
y = [0, 0, 0, 0];
label = ["", "", "", ""];
for epoch = 1:1:100
    text = append(text, sprintf(['Epoch: %d\n' ...
                                '-------------------------------\n'], epoch));
    for p = 1:1:length(x)
        u = w' * x(1:4, p);
        y(p) = (u > 0) * 2 - 1;
        %k = k + 1;
        dw = b * (t(p) - y(p)) * x(1:4, p);
        w = w + dw;

        % Classification labeling
        if y(p) == t(p)
            if y(p) > 0
                label(p) = 'True Positive';
            else
                label(p) = 'True Negative';
            end
        else
            if y(p) > 0
                label(p) = 'False Positive';
            else
                label(p) = 'False Negative';
            end
        end

        % formatting the results
        text = append(text, sprintf('Data Sample %d:\n', p));
        text = append(text, sprintf('\t x%d: [%.2f %.2f %.2f %.2f]^T\n', p, x(1:4, p)));
        text = append(text, sprintf('\t t%d: %+d -> Class: %s\n', p, t(p), class(p)));
        text = append(text, sprintf('\t y%d: %+d -> Label: %s\n', p, y(p), label(p)));
        text = append(text, sprintf('\t Δw%d: [%.2f %.2f %.2f %.2f]^T\n', p, dw));
        text = append(text, sprintf('\t w%d: [%.2f %.2f %.2f %.2f]^T\n', p, w));
    end
    if all(y == t)
        break
    end
end

disp(text)