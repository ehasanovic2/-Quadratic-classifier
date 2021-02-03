%ucitavanje podataka
load fisheriris
%vektor kolona 'species' se sastoji od tri vrste cvijeta iris : setosa,
%versicolor i virginica
%matrica 'meas' se sastoji od mjerenja dužine i širine cašicnih listova (prve dvije kolone) i
%dužine i širine latica cvijeta (druge dvije kolone)
PL = meas(:,3);
PW = meas(:,4);
%plotanje podataka i klasifikacija po vrstama tj. 'species'
figure(1);
h1 = gscatter(PL,PW,species,'krb','ov^',[],'off');
h1(1).LineWidth = 2;
h1(2).LineWidth = 2;
h1(3).LineWidth = 2;
legend('Setosa','Versicolor','Virginica','Location','best')
hold on;


X = [PL,PW];
%kreiranje kvadraticnog klasifikatora
MdlQuadratic = fitcdiscr(X,species,'DiscrimType','quadratic');
qdaClass = resubPredict(MdlQuadratic)
%greška pri pogrešnoj klasifikaciji uzoraka u klase
qdaResubErr = resubLoss(MdlQuadratic)

%racunanje konfuzijske matrice(daje informacije o poznatnoj i predvidjenoj
%klasi uzorka) tj. (i,j) elemenat u matrici predstavljaja broj uzoraka kod
%kojih je poznata klasa i, a predvidjana klasa j, pa elementi na dijagonali
%predstavljaju korektne klasifikacije
[qdaResubCM,grpOrder] = confusionmat(species,qdaClass)
%crtanje x na pogrešno klasificirane uzorke



%isrtavanje podrucja koji pripadaju razlicitim klasama
figure(2);
[x,y] = meshgrid(0:.1:7,0:.1:2.5);
x = x(:);
y = y(:);
j = classify([x y],meas(:,3:4),species);
gscatter(x,y,j,'grb','sod')


%uklanjanje linearnih granica medju klasama
% delete(h2);
% delete(h3);

%odredjivanje koeficjenata za formiranje parabolnicne granice medju klasama
%2 i 3
MdlQuadratic.ClassNames([2 3])
K = MdlQuadratic.Coeffs(2,3).Const;
L = MdlQuadratic.Coeffs(2,3).Linear;
Q = MdlQuadratic.Coeffs(2,3).Quadratic;

figure(3);
h1 = gscatter(PL,PW,species,'krb','ov^',[],'off');
h1(1).LineWidth = 2;
h1(2).LineWidth = 2;
h1(3).LineWidth = 2;
legend('Setosa','Versicolor','Virginica','Location','best')
hold on;
%plotanje parabole izmedju druge i trece klase
f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
    (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h2 = ezplot(f,[.9 7.1 0 2.5]);
h2.Color = 'r';
h2.LineWidth = 2;
hold on;

%odredjivanje koeficjenata za formiranje parabolicne granice izmedju klasa 1
%i 2
MdlQuadratic.ClassNames([1 2])
K = MdlQuadratic.Coeffs(1,2).Const;
L = MdlQuadratic.Coeffs(1,2).Linear;
Q = MdlQuadratic.Coeffs(1,2).Quadratic;

%plotanje parabole izme?u klase 1 i klase 2

f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
    (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h3 = ezplot(f,[.9 7.1 0 1.02]); % Plot the relevant portion of the curve.
h3.Color = 'k';
h3.LineWidth = 2;
axis([.9 7.1 0 2.5])
xlabel('Petal Length')
ylabel('Petal Width')
title('{\bf Quadratic Classification with Fisher Training Data}')
hold on

bad = ~strcmp(qdaClass,species);
hold on;
plot(meas(bad,3), meas(bad,4), 'kx');
hold off;

%kreiranje testnog dijela uzoraka od seta za treniranje jer ne posjedujemo
%jos podataka
%razdvajanje set za treniranje u 10 podgrupa, otprilike iste velicine i
%iste raspodjele podataka u klase kao i orginalni set za treniranje
rng(0,'twister');

%nakon poodjele 9 podskupova podataka je koristeno za treniranje,a jedan za
%testiranje
cp = cvpartition(species,'KFold',10)

%odredjivanje testne greške
cvqda = crossval(MdlQuadratic,'CVPartition',cp);
qdaCVErr = kfoldLoss(cvqda)


