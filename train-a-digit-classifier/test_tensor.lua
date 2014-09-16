require 'torch'

n = 9

v = torch.Tensor(n)

for i = 1,n do
   v[i] = i
end

print(v)

u = v:unfold(1, n, n)
print(u)

print(u:unfold(2, 3, 3))
