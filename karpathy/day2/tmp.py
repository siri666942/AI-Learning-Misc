g=torch.Generator().manual_seed(2147483647)

for i in range(20):
    ix=0
    out=[]
    while True:
        p=P[ix]
        # p=torch.ones(27)
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))






    