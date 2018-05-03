import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.garden import FileBrowser

kivy.require('1.0.6')  # replace with your current kivy version !


class Interface(App):
    def _fbrowser_success(self, instance, TheRoot):
        print(instance.selection)
        TheRoot.dismiss()


if __name__ == '__main__':
    Interface().run()
