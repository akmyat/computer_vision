L = [8; -4; 0]

P = [2, 4, 2 ; 6, 3, 3 ; 1, 2, 0.5 ; 16, 8, 4]

printf("D = P * L\n")
D = P * L

printf("The following points are on line (8, -4, 0)\n");
for i = 1:4
  if D(i, 1) == 0
    printf('(%d, %d)', P(i, 1) / P(i, 3), P(i, 2) / P(i, 3));
  end
end

% Plot line
lineX = linspace(0, 4, 100)
lineY = (-L(1) * lineX - L(3)) / L(2)
plot(lineX, lineY, 'b-')

hold on

% Plot Points
for j = 1:4
  X(j) = P(j, 1) / P(j, 3)
  Y(j) = P(j, 2) / P(j, 3)
end
plot(X, Y, 'r*')