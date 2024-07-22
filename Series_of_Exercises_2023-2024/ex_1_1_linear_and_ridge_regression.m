%                         Μηχανική Μάθηση
%                   1η Σειρά Ασκήσεων 2023-2024
%              Άσκηση 1.1: Linear and Ridge Regression
% -------------------------------------------------------------------------
%                   Χαρίλαος Κουκουλάρης el18137
%                           30/11/2023

% Ανάγνωση των δεδομένων από το αρχείο
wine_data = readtable('ML2023-24-hwk1.csv');

% Κανονικοποίηση δεδομένων
normalized_data = normalize(wine_data);

% Κατανομή τιμών δεδομένων
figure
sgtitle('Value Distribution of Features')
for i = 1:1:11
    subplot(4,3,i)
    hold
    grid
    title(normalized_data(:,i).Properties.VariableNames)
    histogram(wine_data(:,i).Variables)
    hold off
end

% Κανονικοιποιημένη κατανομή τιμών δεδομένων
figure
sgtitle('Normalized Value Distribution of Features')
for i = 1:1:11
    subplot(4,3,i)
    hold
    grid
    title(normalized_data(:,i).Properties.VariableNames)
    histogram(normalized_data(:,i).Variables)
    hold off
end

%% α) ----------------------------------------------------------------------

% Κανονικοποιημένος συντελεστής συσχέτισης μεταξύ στηλών 9 (pH) και 10 (sulphates)
r9_10 = corr(normalized_data(:,9).Variables,normalized_data(:,10).Variables);
r9_10

figure 
corrplot(normalized_data)
%{
[~,~,h] = corrplot(normalized_data);                    %grab handle to plot objects
lineHandles = h(strcmp(get(h.Variables, 'type'), 'line'));       %get handles for scatter plots only
% Loop through each scatter plot
for i = 1:numel(lineHandles)
    x = lineHandles(i).XData;                         %x data 
    y = lineHandles(i).YData;                         %y data
    xlim(lineHandles(i).Parent, [min(x), max(x)]);    % set x limit to range of x data
    ylim(lineHandles(i).Parent, [min(y), max(y)]);    % set y limit to range of y data
    
    % To convince yourself that the axis scales are still the same within rows/cols,
    % include these two lines of code that will display tick marks.
    %lineHandles(i).Parent.Position(3:4) = lineHandles(i).Parent.Position(3:4) * .8; 
    %set(lineHandles(i).Parent, 'XTickMode', 'auto', 'XTickLabelMode', 'auto', 'YTickMode', 'auto', 'YTickLabelMode', 'auto')
end
% now take care of the x axis limits of the histogram plots
histHandles = h(strcmp(get(h, 'type'), 'histogram'));     %handles to all hist plots
% loop through hist plots
for j = 1:numel(histHandles)
    x = histHandles(j).BinEdges;                         %bin edges
    xlim(histHandles(j).Parent, [min(x), max(x)]);       %set x limits
end
%}

% Κανονικοποιημένος πίνακας συσχέτισης δεδομένων
correlation_matrix = corrcoef(normalized_data.Variables);
figure
heatmap(wine_data.Properties.VariableNames, ...
        wine_data.Properties.VariableNames, ...
        correlation_matrix).title('Normalized Correlation Matrix')

% Διαχωρισμός δεδομένων εκπαίδευσης και επαλήθευσης
train_data = normalized_data(1:100,:);
test_data = normalized_data(101:150,:);

% Διαχωρισμός ανεξάρτητων και εξαρτημένης μεταβλητής
X_train = train_data(:,1:end-1).Variables;
y_train = train_data(:,'quality').Variables;

X_test = test_data(:,1:end-1).Variables;
y_test = test_data(:,'quality').Variables;

%% β) ----------------------------------------------------------------------

w_lr = (X_train' * X_train)^(-1) * X_train' * y_train;

%% γ) ----------------------------------------------------------------------

% λ = 10
lambda = 10;

w_r10 = (X_train' * X_train + lambda * eye(11))^(-1) * X_train' * y_train;

% λ = 100
lambda = 100;

w_r100 = (X_train' * X_train + lambda * eye(11))^(-1) * X_train' * y_train;

% λ = 200
lambda = 200;

w_r200 = (X_train' * X_train + lambda * eye(11))^(-1) * X_train' * y_train;

%% δ) ----------------------------------------------------------------------

figure
hold
grid
plot(w_lr, 'o-', 'MarkerFaceColor','auto')
plot(w_r10, 'o-', 'MarkerFaceColor','auto')
plot(w_r100, 'o-', 'MarkerFaceColor','auto')
plot(w_r200, 'o-', 'MarkerFaceColor','auto')
legend('\lambda = 0','\lambda = 10','\lambda = 100','\lambda = 200')
title('Ridge Regression Weights')
ylabel('Weight Value')
xticklabels({'w_1','w_2','w_3','w_4','w_5','w_6','w_7','w_8','w_9','w_{10}','w_{11}'})
xlabel('Feature Weight')

%% ε) ----------------------------------------------------------------------
% sum((X_train * w_lr - y_train).^2) / height(X_train))^0.5
Set = ["Train Set";"Test Set"];
Linear_Regression_lambda_0 = [
    rmse(y_train, X_train * w_lr);
    rmse(y_test, X_test * w_lr)];
Ridge_Regression_lambda_10 = [
    rmse(y_train, X_train * w_r10);
    rmse(y_test, X_test * w_r10)];
Ridge_Regression_lambda_100 = [
    rmse(y_train, X_train * w_r100);
    rmse(y_test, X_test * w_r100)];
Ridge_Regression_lambda_200 = [
    rmse(y_train, X_train * w_r200);
    rmse(y_test, X_test * w_r200)];

results = table(Set, ...
    Linear_Regression_lambda_0, ...
    Ridge_Regression_lambda_10, ...
    Ridge_Regression_lambda_100, ...
    Ridge_Regression_lambda_200, ...
    VariableNames=["RMSE", ...
                   "Linear Regression λ = 0", ...
                   "Ridge Regression λ = 10", ...
                   "Ridge Regression λ = 100", ...
                   "Ridge Regression λ = 200"])

figure
hold
grid
plot(results(1,2:5).Variables, 'o-', 'MarkerFaceColor','auto')
plot(results(2,2:5).Variables, 'o-', 'MarkerFaceColor','auto')
title('Root Mean Squared Error ')
legend('Train Set','Test Set',Location='northwest')
xticks([1 2 3 4])
xticklabels({'\lambda = 0','\lambda = 10','\lambda = 100','\lambda = 200'})


%% Επαλήθευση 
w_lr = ridge(y_train,X_train,0)

w_r10 = ridge(y_train,X_train,10)

w_r100 = ridge(y_train,X_train,100)

w_r200 = ridge(y_train,X_train,200)

Set = ["Train Set";"Test Set"];
Linear_Regression_lambda_0 = [
    rmse(y_train, X_train * w_lr);
    rmse(y_test, X_test * w_lr)];
Ridge_Regression_lambda_10 = [
    rmse(y_train, X_train * w_r10);
    rmse(y_test, X_test * w_r10)];
Ridge_Regression_lambda_100 = [
    rmse(y_train, X_train * w_r100);
    rmse(y_test, X_test * w_r100)];
Ridge_Regression_lambda_200 = [
    rmse(y_train, X_train * w_r200);
    rmse(y_test, X_test * w_r200)];

results = table(Set, ...
    Linear_Regression_lambda_0, ...
    Ridge_Regression_lambda_10, ...
    Ridge_Regression_lambda_100, ...
    Ridge_Regression_lambda_200, ...
    VariableNames=["RMSE", ...
                   "Linear Regression λ = 0", ...
                   "Ridge Regression λ = 10", ...
                   "Ridge Regression λ = 100", ...
                   "Ridge Regression λ = 200"])

figure
hold
grid
plot(results(1,2:5).Variables, 'o-', 'MarkerFaceColor','auto')
plot(results(2,2:5).Variables, 'o-', 'MarkerFaceColor','auto')
title('Root Mean Squared Error ')
legend('Train Set','Test Set',Location='northwest')
xticks([1 2 3 4])
xticklabels({'\lambda = 0','\lambda = 10','\lambda = 100','\lambda = 200'})
