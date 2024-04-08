from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer

from model.DUCK_NET import DUCKNet


class TransDUCKNet(DUCKNet):
    def __init__(self, input_channels, out_classes, starting_filters, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.3):
        super(TransDUCKNet, self).__init__(input_channels, out_classes, starting_filters)
         
        # Transformer Encoder Layer
        self.encoder_layer1 = TransformerEncoderLayer(d_model=starting_filters * 2, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder1 = TransformerEncoder(encoder_layer=self.encoder_layer1, num_layers=num_encoder_layers)
        
        self.encoder_layer2 = TransformerEncoderLayer(d_model=starting_filters * 4, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder2 = TransformerEncoder(encoder_layer=self.encoder_layer2, num_layers=num_encoder_layers)
        
        self.encoder_layer3 = TransformerEncoderLayer(d_model=starting_filters * 8, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder3 = TransformerEncoder(encoder_layer=self.encoder_layer3, num_layers=num_encoder_layers)
        
        self.encoder_layer4 = TransformerEncoderLayer(d_model=starting_filters * 16, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder4 = TransformerEncoder(encoder_layer=self.encoder_layer4, num_layers=num_encoder_layers)
        
        self.encoder_layer5 = TransformerEncoderLayer(d_model=starting_filters * 32, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder5 = TransformerEncoder(encoder_layer=self.encoder_layer5, num_layers=num_encoder_layers)
        
        # Transformer Decoder Layer
        self.decoder_layer1 = TransformerDecoderLayer(d_model=starting_filters * 2, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        
    def forward(self, x):
        # Encoder
        p1 = self.conv1(x)
        p1 = p1.flatten(2).permute(2, 0, 1)  # Transformer를 위한 재구성
        p1 = self.transformer_encoder1(p1)
        p1 = p1.permute(1, 2, 0).view(p1.size(1), -1, int(p1.size(0)**0.5), int(p1.size(0)**0.5))  # 원래 형태로 복구
        
        p2 = self.conv2(p1)
        p2 = p2.flatten(2).permute(2, 0, 1)  # Transformer를 위한 재구성
        p2 = self.transformer_encoder2(p2)
        p2 = p2.permute(1, 2, 0).view(p2.size(1), -1, int(p2.size(0)**0.5), int(p2.size(0)**0.5))  # 원래 형태로 복구
        
        p3 = self.conv3(p2)
        p3 = p3.flatten(2).permute(2, 0, 1)  # Transformer를 위한 재구성
        p3 = self.transformer_encoder3(p3)
        p3 = p3.permute(1, 2, 0).view(p3.size(1), -1, int(p3.size(0)**0.5), int(p3.size(0)**0.5))  # 원래 형태로 복구
        
        p4 = self.conv4(p3)
        p4 = p4.flatten(2).permute(2, 0, 1)  # Transformer를 위한 재구성
        p4 = self.transformer_encoder4(p4)
        p4 = p4.permute(1, 2, 0).view(p4.size(1), -1, int(p4.size(0)**0.5), int(p4.size(0)**0.5))  # 원래 형태로 복구
        
        p5 = self.conv5(p4)
        p5 = p5.flatten(2).permute(2, 0, 1)  # Transformer를 위한 재구성
        p5 = self.transformer_encoder5(p5)
        p5 = p5.permute(1, 2, 0).view(p5.size(1), -1, int(p5.size(0)**0.5), int(p5.size(0)**0.5))  # 원래 형태로 복구

        # 이후 과정은 변하지 않음
        t0 = self.conv_block1(x)
        
        l1i = self.li_conv1(t0)
        s1 = p1 + l1i
        t1 = self.conv_block2(s1)

        l2i = self.conv2(t1)
        s2 = p2 + l2i
        t2 = self.conv_block3(s2)

        l3i = self.conv3(t2)
        s3 = p3 + l3i
        t3 = self.conv_block4(s3)

        l4i = self.conv4(t3)
        s4 = p4 + l4i
        t4 = self.conv_block5(s4)
        
        l5i = self.conv5(t4)
        s5 = p5 + l5i
        t51 = self.conv_block6(s5)
        t53 = self.conv_block7(t51)

        # Decoder
        l5o = self.upconv4(t53)
        c4 = l5o + t4
        q4 = self.conv_block8(c4)

        l4o = self.upconv3(q4)
        c3 = l4o + t3
        q3 = self.conv_block9(c3)

        l3o = self.upconv2(q3)
        c2 = l3o + t2
        q6 = self.conv_block10(c2)

        l2o = self.upconv1(q6)
        c1 = l2o + t1
        q1 = self.conv_block11(c1)

        l1o = self.upconv0(q1)
        c0 = l1o + t0
        z1 = self.conv_block12(c0)

        output = self.final_conv(z1)
        
        return output
