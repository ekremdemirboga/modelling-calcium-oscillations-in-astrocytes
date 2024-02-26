function dy = test(t,y,Xi)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

dy = [
    Xi(2,1)*y(1) + Xi(2,1)*y(2) +  Xi(3,1)*y(3);
    Xi(3,1)*y(1) + Xi(3,1)*y(2) +  Xi(3,1)*y(3);
    Xi(4,1)*y(1) + Xi(4,1)*y(2) +  Xi(4,1)*y(3);
];