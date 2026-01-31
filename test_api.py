"""
API Test Script
Test all endpoints of the mentor-mentee matching API

Usage: python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:5000/api"

def print_response(response, title):
    """Pretty print API response"""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80)
    print(f"Status Code: {response.status_code}")
    
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except:
        print(response.text)

def test_health():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print_response(response, "TEST: Health Check")
    return response.status_code == 200

def test_get_mentors():
    """Test getting all mentors"""
    response = requests.get(f"{BASE_URL}/mentors")
    print_response(response, "TEST: Get Mentors")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Found {data.get('count', 0)} mentors")
        if data.get('mentors'):
            print(f"Sample mentor: {data['mentors'][0].get('name', 'N/A')}")
            return data['mentors'][0]['mentor_id']
    return None

def test_get_mentees():
    """Test getting all mentees"""
    response = requests.get(f"{BASE_URL}/mentees")
    print_response(response, "TEST: Get Mentees")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Found {data.get('count', 0)} mentees")
        if data.get('mentees'):
            print(f"Sample mentee: {data['mentees'][0].get('name', 'N/A')}")
            return data['mentees'][0]['mentee_id']
    return None

def test_recommendations(mentee_id):
    """Test getting recommendations for a mentee"""
    payload = {
        "mentee_id": mentee_id,
        "top_k": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/recommend",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print_response(response, f"TEST: Get Recommendations for Mentee {mentee_id}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Got {data.get('recommendations_count', 0)} recommendations")
        
        if data.get('recommendations'):
            print("\nTop 3 Recommendations:")
            for i, rec in enumerate(data['recommendations'], 1):
                print(f"\n  {i}. {rec['mentor_name']}")
                print(f"     Match Score: {rec['match_score']:.1%}")
                print(f"     Domains: {', '.join(rec['domains'][:3])}")
                print(f"     Domain Overlap: {rec['match_details']['domain_overlap']:.0%}")
            
            return data['recommendations'][0]['mentor_id']
    return None

def test_match_score(mentor_id, mentee_id):
    """Test getting match score for a specific pair"""
    payload = {
        "mentor_id": mentor_id,
        "mentee_id": mentee_id
    }
    
    response = requests.post(
        f"{BASE_URL}/match-score",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    print_response(response, f"TEST: Match Score (Mentor {mentor_id} + Mentee {mentee_id})")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Match Score: {data.get('match_score', 0):.1%}")
        
        if data.get('match_details'):
            details = data['match_details']
            print("\nMatch Details:")
            print(f"  Domain Overlap: {details['domain_overlap']:.0%}")
            print(f"  Skill Overlap: {details['skill_overlap']:.0%}")
            print(f"  Style Match: {'✓' if details['style_match'] else '✗'}")
            print(f"  Timezone Match: {'✓' if details['timezone_match'] else '✗'}")
            print(f"  Availability Compatibility: {details['availability_compatibility']:.0%}")

def main():
    """Run all tests"""
    print("="*80)
    print(" "*25 + "API TEST SUITE")
    print("="*80)
    print("\nMake sure the API is running: python api.py")
    print("\nPress Enter to start tests...")
    input()
    
    try:
        # Test 1: Health Check
        if not test_health():
            print("\n Health check failed. Is the API running?")
            return
        
        print("\n✓ Health check passed!")
        
        # Test 2: Get Mentors
        mentor_id = test_get_mentors()
        if not mentor_id:
            print("\n❌ Failed to get mentors")
            return
        
        # Test 3: Get Mentees
        mentee_id = test_get_mentees()
        if not mentee_id:
            print("\n❌ Failed to get mentees")
            return
        
        # Test 4: Get Recommendations
        recommended_mentor_id = test_recommendations(mentee_id)
        
        # Test 5: Get Match Score
        if recommended_mentor_id:
            test_match_score(recommended_mentor_id, mentee_id)
        elif mentor_id:
            test_match_score(mentor_id, mentee_id)
        
        # Summary
        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED")
        print("="*80)
        print("\nThe API is working correctly!")
        print("\nYou can now:")
        print("  1. Integrate with your frontend")
        print("  2. Deploy to production")
        print("  3. Add authentication and rate limiting")
        
    except requests.exceptions.ConnectionError:
        print("\n" + "="*80)
        print(" CONNECTION ERROR")
        print("="*80)
        print("\nCouldn't connect to the API.")
        print("Make sure the API is running:")
        print("  python api.py")
        print("\nThen run this test script again.")
    
    except Exception as e:
        print(f"\n Test failed with error: {e}")

if __name__ == "__main__":
    main()
