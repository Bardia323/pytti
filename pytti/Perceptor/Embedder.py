def forward(self, diff_image, input, device):
    side_x, side_y = diff_image.shape[2:]
    if input is None:
        input = format_module(diff_image, self).to(device=device)
    else:
        input = format_input(input, diff_image, self).to(device=device)
    
    # Only apply channels_last if tensor has 4 dimensions
    if input.dim() == 4:
        input = input.to(memory_format=torch.channels_last)
    elif input.dim() < 4:
        # Reshape to 4D if needed
        shape = list(input.shape)
        while len(shape) < 4:
            shape.insert(0, 1)  # Add dimensions to make it 4D
        input = input.view(shape).to(memory_format=torch.channels_last)
    
    max_size = min(side_x, side_y)
    image_embeds = []
    # ... existing code ... 