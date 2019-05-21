from primesense import openni2

#OPENNI_REDIST_DIR=
openni2.initialize(r"C:\Users\Administrator\Desktop\oni")

dev = openni2.Device.open_any()

depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()
depth_stream.start()
color_stream.start()
rec = openni2.Recorder("test.oni")
rec.attach(depth_stream)
rec.attach(color_stream)
print (rec.start())

# Do stuff here

rec.stop()
depth_stream.stop()
color_stream.stop()


openni2.unload()