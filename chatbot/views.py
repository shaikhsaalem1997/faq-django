from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from chatbot.models import Message
from chatbot.serializers import MessageSerializer
from .chatbot import ChatBot


chatbot = ChatBot()


def initialize():
    chatbot.download_data()
    chatbot.lemmatize_data()
    chatbot.list_training_data()
    chatbot.train_data()
    print('Training Done....!')


initialize()


@csrf_exempt
def messageApi(request, id=0):
    if request.method == 'GET':
        messages = Message.objects.all()
        messages_serializer = MessageSerializer(messages, many=True)
        return JsonResponse(messages_serializer.data, safe=False)
    elif request.method == 'POST':
        messages_data = JSONParser().parse(request)
        print(' ##### messages_data ::: ', messages_data)
        print(' ##### messages_data ::: ', messages_data['MsgChatIn'])
        messages_serializer = MessageSerializer(data=messages_data)
        print(' ##### messages_serializer ::: ', messages_serializer)
        print(' VALIDITY messages_serializer ::: ', messages_serializer.is_valid())
        if messages_serializer.is_valid():
            messages_serializer.save()
            intents = chatbot.pred_class(messages_data['MsgChatIn'], chatbot.words, chatbot.classes)
            result_chat = chatbot.get_response(intents, chatbot.data)
            print('result_chat====', result_chat)
            return JsonResponse(result_chat, safe=False)
        return JsonResponse("Failed to Add", safe=False)
    elif request.method == 'PUT':
        messages_data = JSONParser().parse(request)
        messages = Message.objects.get(MsgId=messages_data['MsgId'])
        messages_serializer = MessageSerializer(messages, data=messages_data)
        if messages_serializer.is_valid():
            messages_serializer.save()
            return JsonResponse("Updated Successfully", safe=False)
        return JsonResponse("Failed to Update")
    elif request.method == 'DELETE':
        messages = Message.objects.get(MsgId=id)
        messages.delete()
        return JsonResponse("Deleted Successfully", safe=False)