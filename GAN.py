




for epoch in range(num_epoch):
        running_loss = 0

        for i, data in enumerate(trainLoader, 0):
            
            img, _ = data[0], data[1]
            
            #训练判决器
            # img = img.view(batch_size, -1)
            real_img = Variable(img).cuda()
            real_label = Variable(torch.reshape(torch.ones(batch_size), [batch_size, 1])).cuda()
            fake_label = Variable(torch.reshape(torch.zeros(batch_size), [batch_size, 1])).cuda()

            #计算判决器的损失函数
            #识别真图
            real_predict = D(real_img)
            d_loss_real = criterion(real_predict, real_label)
            real_scores = real_predict
            #识别假图
            z = Variable(torch.randn(batch_size, z_dimension)).cuda()
            fake_img = G(z)
            fake_predict = D(fake_img)
            d_loss_fake = criterion(fake_predict, fake_label)
            fake_scores = fake_predict

            #优化判决器
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()


            #训练生成器
            #计算生成器的损失函数
            z = Variable(torch.randn(batch_size, z_dimension)).cuda()
            fake_img = G(z)
            output = D(fake_img)
            g_loss = criterion(output, real_label)

            #生成器优化
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            if (i+1)%1000 == 0:
                print('Epoch [{}/{}]  d_loss: {:.6f},  g_loss: {:.6f},  d_loss_fake: {:.6f}'.format(epoch, num_epoch, drunning_loss/1000, grunning_loss/1000, drunning_loss_fake/1000))
                print('               D real: {:.6f},  D fake: {:.6f}'.format(real_scores.data.mean(), fake_scores.data.mean()))
                drunning_loss = 0
                grunning_loss = 0
                drunning_loss_fake = 0
            else:
                drunning_loss += d_loss.item()
                grunning_loss += g_loss.item()
                drunning_loss_fake += d_loss_fake.item()

        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './img/real_imges.png')
        
        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './img/fake_images-{}.png'.format(epoch+1))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')