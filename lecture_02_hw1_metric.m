clc;clear
%A=[1 2;3 4; 5 6;7 8];
A=[1 2 3 4;8 7 6 5;9 8 7 6;2 3 4 5;1 2 3 4;8 7 6 5];
A1=A*A';
A2=A'*A;
[U1, D1] = eig(A1);
[V1, D2] = eig(A2);

[D1, help] = sort(diag(D1), 'descend');
U=U1(:,help);
D=sqrt(diag(D1));
[D2, help] = sort(diag(D2), 'descend');
V=V1(:,help);
V=V';

AA1=U(:,1)*D(1,1)*V(1,:);
AA2=U(:,1:2)*D(1:2,1:2)*V(1:2,:);

