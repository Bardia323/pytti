def perceive(self, x, angle=0.0):
    """Apply perception kernels to the input tensor"""
    batch_size, c, h, w = x.shape
    
    # Prepare kernels
    identity = self.identity
    dx, dy = self.dx, self.dy
    
    # Apply rotation if needed
    if angle != 0.0:
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        new_dx = c * dx - s * dy
        new_dy = s * dx + c * dy
        dx, dy = new_dx, new_dy
    
    # Fix: Use regular convolution instead of grouped convolution
    identity_kernel = identity.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
    dx_kernel = dx.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
    dy_kernel = dy.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
    
    # Apply convolutions separately and concatenate results
    y_identity = F.conv2d(x, identity_kernel, padding=1, groups=self.channel_n)
    y_dx = F.conv2d(x, dx_kernel, padding=1, groups=self.channel_n)
    y_dy = F.conv2d(x, dy_kernel, padding=1, groups=self.channel_n)
    
    # Concatenate along channel dimension
    y = torch.cat([y_identity, y_dx, y_dy], dim=1)
    
    return y 