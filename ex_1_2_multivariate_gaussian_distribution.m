%                         Μηχανική Μάθηση
%                   1η Σειρά Ασκήσεων 2023-2024
%          Άσκηση 1.2: Multivariate Gaussian Distribution
% -------------------------------------------------------------------------
%                   Χαρίλαος Κουκουλάρης el18137
%                           6/12/2023

mi = [-2; 0; 2];
Sigma = [1 0.8 0.5; 0.8 2 1; 0.5 1 3];

%% Ερώτημα β. --------------------------------------------------------------

% Επιλογή των τυχαίων μεταβλητών x1 και x2 με x3 = 1
a = [1 2];
b = 3;
xb = 1;

% Σa|b = Σaa - Σab Σbb^-1 Σba
Sigma1 = Sigma(a,a) - Sigma(a,b) * Sigma(b,b)^-1 * Sigma(b,a)

% μa|b = μa + Σab Σbb^-1 (xb - μb)
mi1 = mi(a) + Sigma(a,b) * Sigma(b,b)^-1 * (xb - mi(b))

% Πλέγμα για την απεικόνιση της κατανομής πιθανότητας
x1 = [mi1(1)-3:0.2:mi1(1)+3 ; mi1(2)-3:0.2:mi1(2)+3];
[X1, X2] = meshgrid(x1(1,:),x1(2,:));
X = [X1(:) X2(:)];

% Υπο συνθήκη συνάρτηση πυκνότητας πιθανότητας p(x1,x2|x3=1)
y1 = mvnpdf(X,mi1',Sigma1);
y1 = reshape(y1,length(x1(1,:)),length(x1(2,:)));

figure
surf(x1(1,:),x1(2,:),y1)
title('Probability Density')
xlabel('x_1')
ylabel('x_2')
zlabel('Probability Density')

%% Ερώτημα γ. --------------------------------------------------------------

% Επιλογή των τυχαίων μεταβλητών x1 και x3 με x2 = 1
a = [1 3];
b = 2;
xb = 1;

% Σa|b = Σaa - Σab Σbb^-1 Σba
Sigma2 = Sigma(a,a) - Sigma(a,b) * Sigma(b,b)^-1 * Sigma(b,a)

% μa|b = μa + Σab Σbb^-1 (xb - μb)
mi2 = mi(a) + Sigma(a,b) * Sigma(b,b)^-1 * (xb - mi(b))

% Πλέγμα για την απεικόνιση της κατανομής πιθανότητας
x2 = [mi2(1)-3:0.2:mi2(1)+3 ; mi2(2)-3:0.2:mi2(2)+3];
[X1, X2] = meshgrid(x2(1,:),x2(2,:));
X = [X1(:) X2(:)];

% Υπο συνθήκη συνάρτηση πυκνότητας πιθανότητας p(x1,x3|x2=1)
y2 = mvnpdf(X,mi2',Sigma2);
y2 = reshape(y2,length(x2(1,:)),length(x2(2,:)));

figure
surf(x2(1,:),x2(2,:),y2)
title('Probability Density')
xlabel('x_1')
ylabel('x_2')
zlabel('Probability Density')

%% Ερώτημα δ. --------------------------------------------------------------

% Ισοσταθμικές καμπύλες των κατανομών πιθανότητας των ερωτημάτων β και γ
figure
hold
title('Contour Lines of Probability Distributions')
contour(x1(1,:),x1(2,:),y1)
contour(x2(1,:),x2(2,:),y2,'--')
legend('p(x_1,x_2|x_3 = 1)','p(x_1,x_3|x_2 = 1)')
xlabel('x')
ylabel('y')