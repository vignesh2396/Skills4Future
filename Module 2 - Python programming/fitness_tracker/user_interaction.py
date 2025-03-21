def view_workouts(user):
    print(f"{user.name}'s Workouts:")
    user.view_workouts()

def save_data(user):
    filename = input("Enter the filename to save data: ")
    user.save_data(filename)
    print("Data saved successfully!")

def load_data(user):
    filename = input("Enter the filename to load data: ")
    user.load_data(filename)
    print("Data loaded successfully!")