import pickle
import os

def manage_faces():
    if not os.path.exists("face_encodings.pkl"):
        print("No face encodings found!")
        return
    
    # Load encodings
    with open("face_encodings.pkl", "rb") as f:
        encodings = pickle.load(f)
    
    if not encodings:
        print("No faces saved!")
        return
    
    while True:
        print("\n" + "=" * 50)
        print("SAVED FACES:")
        print("=" * 50)
        
        names = list(encodings.keys())
        for i, name in enumerate(names, 1):
            print(f"{i}. {name}")
        
        print("\n" + "=" * 50)
        print("OPTIONS:")
        print("=" * 50)
        print("Enter number to DELETE that person")
        print("Type 'all' to delete ALL faces")
        print("Type 'q' to quit")
        print("=" * 50)
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'all':
            confirm = input("⚠️  Delete ALL faces? (yes/no): ").strip().lower()
            if confirm == 'yes':
                os.remove("face_encodings.pkl")
                print("✅ All faces deleted!")
                break
            else:
                print("❌ Cancelled")
        elif choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(names):
                name_to_delete = names[num - 1]
                confirm = input(f"⚠️  Delete {name_to_delete}? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    del encodings[name_to_delete]
                    
                    # Save updated encodings
                    with open("face_encodings.pkl", "wb") as f:
                        pickle.dump(encodings, f)
                    
                    print(f"✅ {name_to_delete} deleted!")
                    
                    if not encodings:
                        print("No more faces saved. Deleting file...")
                        os.remove("face_encodings.pkl")
                        break
                else:
                    print("❌ Cancelled")
            else:
                print("❌ Invalid number!")
        else:
            print("❌ Invalid choice!")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    manage_faces()