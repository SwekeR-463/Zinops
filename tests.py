import numpy as np
from zinops import rearrange

def tests():
    x = np.random.rand(3, 4)
    result = rearrange(x, 'h w -> w h')
    print(result.shape)
    assert result.shape == (4, 3)
    
    x = np.random.rand(12, 10)
    result = rearrange(x, '(h w) c -> h w c', h=3)
    print(result.shape)
    assert result.shape == (3, 4, 10)
    
    x = np.random.rand(3, 4, 5)
    result = rearrange(x, 'a b c -> (a b) c')
    print(result.shape)
    assert result.shape == (12, 5)
    
    x = np.random.rand(3, 1, 5)
    result = rearrange(x, 'a 1 c -> a b c', b=4)
    print(result.shape)
    assert result.shape == (3, 4, 5)
    
    x = np.random.rand(2, 3, 4, 5)
    result = rearrange(x, '... h w -> ... (h w)')
    print(result.shape)
    assert result.shape == (2, 3, 20)
    
    x = np.random.rand(24,)
    result = rearrange(x, '(a b) -> a b', a=4, b=6)
    print(result.shape)
    assert result.shape == (4, 6)
    
    x = np.random.rand(12, 4)
    result = rearrange(x, '(h1 h2) w -> h1 (h2 w)', h1=3)
    print(result.shape)
    assert result.shape == (3, 16)
    
    images = np.random.randn(30, 40, 3, 32)
    r = rearrange(images, 'b h w c -> h (b w) c')
    print(r.shape)
    assert r.shape == (40, 90, 32)
    
    images = np.random.randn(30, 40, 3, 32)
    r = rearrange(images, 'b h w c -> b (c h w)')
    print(r.shape)
    assert r.shape == (30, 3840)


if __name__ == '__main__':
    tests()