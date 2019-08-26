%% The objective function
% function f = Obj(d, t, z, A, b, L)
% x=z(1:d,1);
% y=z(d+1:2*d,1);
% %t=z(end,1);
% f1 =y;
% f2 = -(5/t)*y -4* ((A'*A)*x - A'*b + L*x);
% f3 =1;
% f=[f1;f2;f3];
% end
%
function f = Obj(d, t, z, A, b, L, p)
x=z(1:d,1);
y=z(d+1:2*d,1);
tau = b .* (A * x);
%t=z(end,1);
f1 = y;
f2 = -((2*p+1)/t)*y - (p^2)* (A' * (- b ./ (1  + exp(tau))) +  L * x);
f3 =1;
f=[f1;f2;f3];
end



