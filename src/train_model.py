def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch_index):
    
    running_loss = 0
    last_loss = 0

    model.train()
    for batch_index, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_index % 20 == 19:
            last_loss = running_loss / 20
            running_loss = 0
            print(f'Эпоха: {epoch_index}, номер батча: {batch_index}, ошибка: {last_loss}')
    return last_loss