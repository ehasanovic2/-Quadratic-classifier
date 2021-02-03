%u?itavanje podataka
load fisheriris
%vektor kolona 'species' se sastoji od tri vrste cvijeta iris : setosa,
%versicolor i virginica
%matrica 'meas' se sastoji od mjerenja dužine i širine ?aši?nih listova (prve dvije kolone) i
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
%hold on

%kreiranje linearnog klasifikatora
X = [PL,PW];
MdlLinear = fitcdiscr(X,species);
ldaClass = resubPredict(MdlLinear);
%greška pri pogrešnoj klasifikaciji uzoraka u klase
ldaResubErr = resubLoss(MdlLinear)

%racunanje konfuzijske matrice(daje informacije o poznatnoj i predvidjenoj
%klasi uzorka) tj. (i,j) elemenat u matrici predstavljaja broj uzoraka kod
%kojih je poznata klasa i, a predvidjana klasa j, pa elementi na dijagonali
%predstavljaju korektne klasifikacije
[ldaResubCM,grpOrder] = confusionmat(species,ldaClass)

figure(2)
%isrtavanje podrucja koji pripadaju razlicitim klasama
[x,y] = meshgrid(0:.1:7,0:.1:2.5);
x = x(:);
y = y(:);
j = classify([x y],meas(:,3:4),species);
gscatter(x,y,j,'grb','sod')


%odredjivanje koeficjenata za linearnu granicu izmedju druge i trece klase
MdlLinear.ClassNames([2 3])
K = MdlLinear.Coeffs(2,3).Const;
L = MdlLinear.Coeffs(2,3).Linear;

figure(3)
h1 = gscatter(PL,PW,species,'krb','ov^',[],'off');
h1(1).LineWidth = 2;
h1(2).LineWidth = 2;
h1(3).LineWidth = 2;
legend('Setosa','Versicolor','Virginica','Location','best')
hold on
%plotanje linije izmedju druge i trece klase
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[.9 7.1 0 2.5]);
h2.Color = 'r';
h2.LineWidth = 2;

%odredjivanje koeficjenata za linearnu granicu izmedju prve i druge klase
MdlLinear.ClassNames([1 2])
K = MdlLinear.Coeffs(1,2).Const;
L = MdlLinear.Coeffs(1,2).Linear;

%plotanje linije koja razdvaja prvu i drugu klasu
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h3 = ezplot(f,[.9 7.1 0 2.5]);
h3.Color = 'k';
h3.LineWidth = 2;
axis([.9 7.1 0 2.5])
xlabel('Petal Length')
ylabel('Petal Width')
title('{\bf Linear Classification with Fisher Training Data}')
hold on;

bad = ~strcmp(ldaClass,species);
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
cvlda = crossval(MdlLinear,'CVPartition',cp);
ldaCVErr = kfoldLoss(cvlda)

