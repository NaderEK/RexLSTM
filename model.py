from vision_lstm.VisionLSTM import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class RexLSTM2(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], channels=[48, 96, 192, 384], num_refinement=4, conv_type="causal1d",
                 expansion_factor=2.66):
        super(RexLSTM2, self).__init__()

        #self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        n_filter_list = (3, 48, 96)
        upsample_stem_list = (96, 48, 3)

        self.conv_stem = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels = n_filter_list[i],
                          out_channels = n_filter_list[i + 1],
                          kernel_size=3,  
                          stride=2, 
                          padding=1), 
            )
                for i in range(len(n_filter_list)-1)
            ])
                
        self.conv_stem.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1], 
                                            out_channels = channels[0], 
                                            stride=1, 
                                            kernel_size=1, 
                                            padding=0)
        )

        self.upsample_stem = nn.Sequential(
            *[nn.Sequential(
                nn.ConvTranspose2d(in_channels=upsample_stem_list[i],
                                   out_channels=upsample_stem_list[i + 1],
                                   kernel_size=3,  
                                   stride=2, 
                                   padding=1, 
                                   output_padding=1)  # Ensures correct spatial size
            )
            for i in range(len(upsample_stem_list)-1)]
        )
        
        self.upsample_stem.add_module("conv_1x1", nn.Conv2d(
            in_channels=upsample_stem_list[-1], 
            out_channels=3, 
            kernel_size=1,  
            stride=1,  
            padding=0)
        )


        self.encoders = nn.ModuleList([nn.Sequential(*[ViLBlockPair(
                        dim=num_ch, conv_kind=conv_type, num_blocks=1) for _ in range(num_tb)])
                                       for num_tb, num_ch in zip(num_blocks,  channels)])
        # the number of downsample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
         # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                              for i in reversed(range(2, len(channels)))])
         # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[ViLBlockPair(dim=channels[2], conv_kind="causal1d", num_blocks=1)
                                                for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[ViLBlockPair(dim=channels[1], conv_kind="causal1d", num_blocks=1)
                                      for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[ViLBlockPair(dim=channels[1], conv_kind="causal1d", num_blocks=1)
                                            for _ in range(num_blocks[0])]))
        self.refinement = nn.Sequential(*[ViLBlockPair(dim=channels[1], conv_kind="causal1d", num_blocks=1)
                                  for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.conv_stem(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.upsample_stem(fr) + x
        return out


class RexLSTM(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 4], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], conv_type="causal1d",
                 num_refinement=4, expansion_factor=2.66):
        super(RexLSTM, self).__init__()
                     
        directions = []
        dpr = [0] * 12
                     
        for i in range(12):
            if i % 2 == 0:
                directions.append(SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
            else:
                directions.append(SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)
        
        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks[:-1], num_heads[:-1], channels[:-1])])
        
        # using x-lstm block
        self.encoders.append(nn.Sequential(*[ViLBlock(
                    dim=channels[-1],
                    drop_path=dpr[i],
                    direction=directions[i],
                )
                for i in range(12)
            ]))
        
        # the number of downsample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # The channel of the last one has not changed
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out
